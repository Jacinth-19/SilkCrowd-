#include "hair_gpu_renderer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

#include "silkcrowd/image.h"
#include "silkcrowd/root_placement.h"

namespace silk {
namespace {

uint16_t toHalf(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant |= 0x800000u;
        const int32_t shift = 14 - exp;
        uint32_t m = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) m += 1u;
        return static_cast<uint16_t>(sign | m);
    }
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    uint32_t m = mant >> 13;
    if (mant & 0x1000u) m += 1u;
    if (m & 0x0400u) { m = 0; ++exp; if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u); }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (m & 0x03FFu));
}

// ---------------------------------------------------------------------------
//  std140 uniform blocks (must mirror hair_common.glsl / composite.frag)
// ---------------------------------------------------------------------------
struct HairModelUniforms {
    float model[16];
    float modelViewProj[16];
    float modelInv[16];
    float lightModelViewProj[16];
    float textureSize[4];
    float blocks[4];
    float cameraPos[4];
    float viewportSize[4];
    float lodParams[4];
    float lodScreen[4];
    float styling0[4];
    float styling1[4];
    float styling4[4];
    float styling2[4];
    float styling3[4];
    float hairColor[4];
    float hairParams[4];
};

struct CameraUniforms {
    float view[16];
    float proj[16];
    float cameraPos[4];
    float sunDirection[4];
    float sunColor[4];
    float skyZenith[4];
    float skyHorizon[4];
    float lodScreen[4];
};

// CompositeUniforms is declared in hair_gpu_renderer.h so the editor can fill it too.

struct LightUniforms {
    float lightViewProj[16];
    float shadowParams[4];
    float hairParams[4];
};

struct HairPush {
    uint32_t chunksPerBundle;
    uint32_t tasksPerChunk;
    uint32_t frameIndex;
    uint32_t flags;
    float shadeColor[4];
    float shadeParams[4];
    float shadeParams2[4];
};

struct HeadPush {
    float mvp[16];
    float model[16];
    float albedo[4];  // rgb + specular/skin flag
};

void copyMat(const Mat4& m, float out[16]) {
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) out[r * 4 + c] = m.m[r][c];
}

}  // namespace

// =============================================================================
//  Pipeline creation helpers
// =============================================================================
bool HairGpuRenderer::writeSampleDescriptors() {
    VkDevice dev = ctx_->device();
    VkDescriptorBufferInfo infos[7];
    VkWriteDescriptorSet writes[7]{};
    VkBuffer buffers[7] = {sampleBuffer_, sampleTriangleBuffer_, workItemBuffer_,
                           strandVertexBuffer_, strandTangentBuffer_, strandIndexBuffer_,
                           strandMetaBuffer_};
    for (int i = 0; i < 7; ++i) {
        infos[i] = {buffers[i], 0, VK_WHOLE_SIZE};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = sampleDescriptorSet_;
        writes[i].dstBinding = static_cast<uint32_t>(i);
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(dev, 7, writes, 0, nullptr);
    return true;
}

namespace {

// Every pass sets its viewport and scissor with vkCmdSetViewport/vkCmdSetScissor, so the
// pipelines must declare them dynamic: with a static VkPipelineViewportStateCreateInfo and
// no VkPipelineDynamicStateCreateInfo, pViewports/pScissors stay null, which is invalid usage
// and crashed lavapipe inside vkCreateGraphicsPipelines (CI, x86_64).
VkPipelineDynamicStateCreateInfo dynamicViewportScissor() {
    static const VkDynamicState states[] = {VK_DYNAMIC_STATE_VIEWPORT,
                                            VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{};
    dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = states;
    return dynamic;
}

// VkSampleCountFlagBits values are bit masks, so they AND directly with the
// framebufferColorSampleCounts / framebufferDepthSampleCounts device limits.
VkSampleCountFlagBits sampleCountFlag(int samples) {
    switch (samples) {
        case 2: return VK_SAMPLE_COUNT_2_BIT;
        case 4: return VK_SAMPLE_COUNT_4_BIT;
        case 8: return VK_SAMPLE_COUNT_8_BIT;
        case 16: return VK_SAMPLE_COUNT_16_BIT;
        default: return VK_SAMPLE_COUNT_1_BIT;
    }
}

}  // namespace

// The three pipelines that write into the multisampled HDR pass. They are kept apart
// from createPipelines() because the MSAA sample count can change at run time (the
// editor's 8x MSAA toggle): changing it only rebuilds these pipelines and the MSAA
// targets, not the layouts, descriptor sets or the single-sample passes (shadow, bloom,
// composite, TAA).
bool HairGpuRenderer::createMainPassPipelines(std::string* error) {
    VkDevice dev = ctx_->device();

    // ---- hair: task + mesh shader pipeline (preferred) ----
    if (hasMeshShaders_) {
        VkShaderModule task = loadShader("hair.task.spv", error);
        VkShaderModule mesh = loadShader("hair.mesh.spv", error);
        VkShaderModule frag = loadShader("hair.frag.spv", error);
        if (task == VK_NULL_HANDLE || mesh == VK_NULL_HANDLE || frag == VK_NULL_HANDLE) return false;

        VkPipelineShaderStageCreateInfo stages[3]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_TASK_BIT_EXT;
        stages[0].module = task;
        stages[0].pName = "main";
        stages[1] = stages[0];
        stages[1].stage = VK_SHADER_STAGE_MESH_BIT_EXT;
        stages[1].module = mesh;
        stages[2] = stages[0];
        stages[2].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[2].module = frag;

        VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        VkFormat colorFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachmentFormats = &colorFormat;
        rendering.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;

        VkPipelineColorBlendAttachmentState blend{};
        blend.blendEnable = VK_TRUE;             // hair is composited source-over
        blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                               VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

        VkPipelineColorBlendStateCreateInfo blendState{
            VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        blendState.attachmentCount = 1;
        blendState.pAttachments = &blend;

        VkPipelineDepthStencilStateCreateInfo depth{
            VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        depth.depthTestEnable = VK_TRUE;
        depth.depthWriteEnable = VK_FALSE;       // hair is transparent: blend, no depth write
        depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = sampleCountFlag(msaaSamples_);

        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp.viewportCount = 1;
        vp.scissorCount = 1;
        VkPipelineVertexInputStateCreateInfo emptyInput{
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};

        VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        gp.pNext = &rendering;
        gp.stageCount = 3;
        gp.pStages = stages;
        gp.pVertexInputState = &emptyInput;
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        gp.pInputAssemblyState = &ia;
        gp.pViewportState = &vp;
        VkPipelineDynamicStateCreateInfo dynamic = dynamicViewportScissor();
        gp.pDynamicState = &dynamic;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.lineWidth = 1.0f;
        gp.pRasterizationState = &rs;
        gp.pMultisampleState = &ms;
        gp.pDepthStencilState = &depth;
        gp.pColorBlendState = &blendState;
        gp.layout = hairPipelineLayout_;
        if (vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp, nullptr, &hairMeshPipeline_) !=
            VK_SUCCESS) {
            if (error) *error = "failed to create the task/mesh hair pipeline";
            return false;
        }
        // Shadow variant of the same pipeline (depth only, light matrix). The shadow map is
        // single-sampled, unlike the HDR pass this pipeline was built for.
        VkShaderModule shadowFrag = loadShader("shadow.frag.spv", error);
        if (shadowFrag == VK_NULL_HANDLE) return false;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        stages[2].module = shadowFrag;
        rendering.colorAttachmentCount = 0;
        rendering.pColorAttachmentFormats = nullptr;
        rendering.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;
        depth.depthWriteEnable = VK_TRUE;
        depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        blend.blendEnable = VK_FALSE;
        if (vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp, nullptr, &shadowHairPipeline_) !=
            VK_SUCCESS) {
            if (error) *error = "failed to create the hair shadow pipeline";
            return false;
        }
        vkDestroyShaderModule(dev, task, nullptr);
        vkDestroyShaderModule(dev, mesh, nullptr);
        vkDestroyShaderModule(dev, frag, nullptr);
        vkDestroyShaderModule(dev, shadowFrag, nullptr);
    }

    // ---- vertex-path hair pipeline (consumes the compute output) ----
    {
        VkShaderModule vert = loadShader("hair_draw.vert.spv", error);
        VkShaderModule frag = loadShader("hair.frag.spv", error);
        if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE) return false;
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vert;
        stages[0].pName = "main";
        stages[1] = stages[0];
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = frag;

        VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        VkFormat colorFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachmentFormats = &colorFormat;
        rendering.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;

        VkPipelineColorBlendAttachmentState blend{};
        blend.blendEnable = VK_TRUE;
        blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blend.colorWriteMask = 0xF;
        VkPipelineColorBlendStateCreateInfo blendState{
            VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        blendState.attachmentCount = 1;
        blendState.pAttachments = &blend;
        VkPipelineDepthStencilStateCreateInfo depth{
            VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        depth.depthTestEnable = VK_TRUE;
        depth.depthWriteEnable = VK_FALSE;
        depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = sampleCountFlag(msaaSamples_);
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp.viewportCount = 1;
        vp.scissorCount = 1;
        VkPipelineVertexInputStateCreateInfo emptyInput{
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.lineWidth = 1.0f;

        VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        gp.pNext = &rendering;
        gp.stageCount = 2;
        gp.pStages = stages;
        gp.pVertexInputState = &emptyInput;
        gp.pInputAssemblyState = &ia;
        gp.pViewportState = &vp;
        VkPipelineDynamicStateCreateInfo dynamic = dynamicViewportScissor();
        gp.pDynamicState = &dynamic;
        gp.pRasterizationState = &rs;
        gp.pMultisampleState = &ms;
        gp.pDepthStencilState = &depth;
        gp.pColorBlendState = &blendState;
        gp.layout = drawPipelineLayout_;
        if (vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp, nullptr, &hairDrawPipeline_) !=
            VK_SUCCESS) {
            if (error) *error = "failed to create the compute-path draw pipeline";
            return false;
        }
        // Depth-only twin: the same ribbons rasterised into the single-sampled shadow map.
        // Without it the compute fallback (Adreno, Intel UHD 630) would cast no hair shadow.
        VkShaderModule shadowFrag = loadShader("shadow.frag.spv", error);
        if (shadowFrag == VK_NULL_HANDLE) return false;
        stages[1].module = shadowFrag;
        rendering.colorAttachmentCount = 0;
        rendering.pColorAttachmentFormats = nullptr;
        rendering.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;
        blend.blendEnable = VK_FALSE;
        depth.depthWriteEnable = VK_TRUE;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        if (vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp, nullptr,
                                      &hairShadowDrawPipeline_) != VK_SUCCESS) {
            if (error) *error = "failed to create the compute-path shadow pipeline";
            return false;
        }
        vkDestroyShaderModule(dev, vert, nullptr);
        vkDestroyShaderModule(dev, frag, nullptr);
        vkDestroyShaderModule(dev, shadowFrag, nullptr);
    }

    // ---- head / teapot ----
    {
        VkShaderModule vert = loadShader("head.vert.spv", error);
        VkShaderModule frag = loadShader("head.frag.spv", error);
        if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE) return false;
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vert;
        stages[0].pName = "main";
        stages[1] = stages[0];
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = frag;

        VkVertexInputBindingDescription binding{0, 8 * sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX};
        VkVertexInputAttributeDescription attrs[3] = {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},
                                                     {1, 0, VK_FORMAT_R32G32B32_SFLOAT, 12},
                                                     {2, 0, VK_FORMAT_R32G32_SFLOAT, 24}};
        VkPipelineVertexInputStateCreateInfo input{
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        input.vertexBindingDescriptionCount = 1;
        input.pVertexBindingDescriptions = &binding;
        input.vertexAttributeDescriptionCount = 3;
        input.pVertexAttributeDescriptions = attrs;

        VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        VkFormat colorFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachmentFormats = &colorFormat;
        rendering.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;

        VkPipelineColorBlendAttachmentState blend{};
        blend.blendEnable = VK_FALSE;
        blend.colorWriteMask = 0xF;
        VkPipelineColorBlendStateCreateInfo blendState{
            VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        blendState.attachmentCount = 1;
        blendState.pAttachments = &blend;
        VkPipelineDepthStencilStateCreateInfo depth{
            VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        depth.depthTestEnable = VK_TRUE;
        depth.depthWriteEnable = VK_TRUE;
        depth.depthCompareOp = VK_COMPARE_OP_LESS;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = sampleCountFlag(msaaSamples_);
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp.viewportCount = 1;
        vp.scissorCount = 1;
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_BACK_BIT;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth = 1.0f;

        VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        gp.pNext = &rendering;
        gp.stageCount = 2;
        gp.pStages = stages;
        gp.pVertexInputState = &input;
        gp.pInputAssemblyState = &ia;
        gp.pViewportState = &vp;
        VkPipelineDynamicStateCreateInfo dynamic = dynamicViewportScissor();
        gp.pDynamicState = &dynamic;
        gp.pRasterizationState = &rs;
        gp.pMultisampleState = &ms;
        gp.pDepthStencilState = &depth;
        gp.pColorBlendState = &blendState;
        gp.layout = headPipelineLayout_;
        if (vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp, nullptr, &headPipeline_) !=
            VK_SUCCESS) {
            if (error) *error = "failed to create the head pipeline";
            return false;
        }
        vkDestroyShaderModule(dev, vert, nullptr);
        vkDestroyShaderModule(dev, frag, nullptr);
    }

    return true;
}

void HairGpuRenderer::destroyMainPassPipelines() {
    VkDevice dev = ctx_->device();
    if (dev == VK_NULL_HANDLE) return;
    VkPipeline* pipelines[] = {&hairMeshPipeline_, &hairDrawPipeline_, &hairShadowDrawPipeline_,
                               &headPipeline_};
    for (VkPipeline* p : pipelines) {
        if (*p != VK_NULL_HANDLE) vkDestroyPipeline(dev, *p, nullptr);
        *p = VK_NULL_HANDLE;
    }
}

bool HairGpuRenderer::createPipelines(std::string* error) {
    VkDevice dev = ctx_->device();

    // ---- hair pipeline layout: shared by the task/mesh path and the compute fallback ----
    {
        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_ALL;
        pc.size = 160;   // HairPush (both paths) plus the vertex path's matrices
        VkDescriptorSetLayout sets[5] = {frameLayout_, modelLayout_, bundleLayout_,
                                         sampleLayout_, hairTextureLayout_};
        VkPipelineLayoutCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        ci.setLayoutCount = 5;
        ci.pSetLayouts = sets;
        ci.pushConstantRangeCount = 1;
        ci.pPushConstantRanges = &pc;
        SILK_VK_CHECK(vkCreatePipelineLayout(dev, &ci, nullptr, &hairPipelineLayout_));
    }

    {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = descriptorPool_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &sampleLayout_;
        SILK_VK_CHECK(vkAllocateDescriptorSets(dev, &ai, &sampleDescriptorSet_));
    }
    if (!writeSampleDescriptors()) return false;

    // ---- composite uniform buffer + descriptor sets ----
    compositeUniform_ = ctx_->createBuffer(sizeof(CompositeUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                           &compositeUniformMemory_);
    if (compositeUniform_ == VK_NULL_HANDLE) return false;
    vkMapMemory(dev, compositeUniformMemory_, 0, sizeof(CompositeUniforms), 0, &compositeUniformMapped_);
    {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = descriptorPool_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &compositeLayout_;
        SILK_VK_CHECK(vkAllocateDescriptorSets(dev, &ai, &compositeSet_));
        VkDescriptorImageInfo imgs[2];
        imgs[0] = {samplerLinear_, hdrView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        imgs[1] = {samplerLinear_, bloomView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorBufferInfo ubo{compositeUniform_, 0, sizeof(CompositeUniforms)};
        VkWriteDescriptorSet writes[3]{};
        for (int i = 0; i < 2; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = compositeSet_;
            writes[i].dstBinding = static_cast<uint32_t>(i);
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[i].pImageInfo = &imgs[i];
        }
        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = compositeSet_;
        writes[2].dstBinding = 2;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[2].pBufferInfo = &ubo;
        vkUpdateDescriptorSets(dev, 3, writes, 0, nullptr);

        VkDescriptorSetAllocateInfo bai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        bai.descriptorPool = descriptorPool_;
        bai.descriptorSetCount = 1;
        bai.pSetLayouts = &bloomLayout_;
        SILK_VK_CHECK(vkAllocateDescriptorSets(dev, &bai, &bloomSet_));
        VkDescriptorImageInfo bloomSrc{samplerLinear_, hdrView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet bw{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        bw.dstSet = bloomSet_;
        bw.dstBinding = 0;
        bw.descriptorCount = 1;
        bw.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bw.pImageInfo = &bloomSrc;
        vkUpdateDescriptorSets(dev, 1, &bw, 0, nullptr);
    }

    // ---- TAA resolve pipeline layout (its descriptor set layout is created above) ----
    {
        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pc.size = 16;
        VkPipelineLayoutCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        ci.setLayoutCount = 1;
        ci.pSetLayouts = &taaLayout_;
        ci.pushConstantRangeCount = 1;
        ci.pPushConstantRanges = &pc;
        SILK_VK_CHECK(vkCreatePipelineLayout(dev, &ci, nullptr, &taaPipelineLayout_));
    }

    // ---- head pipeline layout (push constants only) ----
    {
        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pc.size = sizeof(HeadPush);
        VkDescriptorSetLayout sets[1] = {frameLayout_};
        VkPipelineLayoutCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        ci.setLayoutCount = 1;
        ci.pSetLayouts = sets;
        ci.pushConstantRangeCount = 1;
        ci.pPushConstantRanges = &pc;
        SILK_VK_CHECK(vkCreatePipelineLayout(dev, &ci, nullptr, &headPipelineLayout_));
    }
    // ---- composite pipeline layout (the descriptor set layout is created in
    // createDescriptorLayouts(), because the composite set is allocated earlier) ----
    {

        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pc.size = 16;
        VkPipelineLayoutCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        ci.setLayoutCount = 1;
        ci.pSetLayouts = &compositeLayout_;
        ci.pushConstantRangeCount = 1;
        ci.pPushConstantRanges = &pc;
        SILK_VK_CHECK(vkCreatePipelineLayout(dev, &ci, nullptr, &compositePipelineLayout_));
    }
    // bloomLayout_ is created in createDescriptorLayouts() as well.
    // ---- compute-fallback draw layout (vertex shader reads the SSBOs) ----
    {
        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_ALL;
        pc.size = 160;   // the vertex path's mat4 mvp + mat4 model + params, and HairPush
        VkDescriptorSetLayout sets[5] = {frameLayout_, modelLayout_, bundleLayout_,
                                         sampleLayout_, hairTextureLayout_};
        VkPipelineLayoutCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        ci.setLayoutCount = 5;
        ci.pSetLayouts = sets;
        ci.pushConstantRangeCount = 1;
        ci.pPushConstantRanges = &pc;
        SILK_VK_CHECK(vkCreatePipelineLayout(dev, &ci, nullptr, &drawPipelineLayout_));
    }

    // ---- pipelines that render into the (possibly multisampled) HDR pass ----
    if (!createMainPassPipelines(error)) return false;

    // ---- compute fallback ----
    {
        VkShaderModule comp = loadShader("hair_compute.comp.spv", error);
        if (comp == VK_NULL_HANDLE) return false;
        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = comp;
        stage.pName = "main";
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage = stage;
        ci.layout = hairPipelineLayout_;
        if (vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr, &hairComputePipeline_) !=
            VK_SUCCESS) {
            if (error) *error = "failed to create the hair compute pipeline";
            return false;
        }
        vkDestroyShaderModule(dev, comp, nullptr);
    }

    // ---- bloom + composite (fullscreen) ----
    {
        VkShaderModule vert = loadShader("fullscreen.vert.spv", error);
        if (vert == VK_NULL_HANDLE) return false;
        auto makeFullscreen = [&](const char* fs, VkFormat format, bool blendOn,
                                  VkPipelineLayout layout, VkPipeline* out) -> bool {
            VkShaderModule frag = loadShader(fs, error);
            if (frag == VK_NULL_HANDLE) return false;
            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = vert;
            stages[0].pName = "main";
            stages[1] = stages[0];
            stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = frag;
            VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
            rendering.colorAttachmentCount = 1;
            rendering.pColorAttachmentFormats = &format;
            VkPipelineColorBlendAttachmentState blend{};
            blend.blendEnable = blendOn ? VK_TRUE : VK_FALSE;
            blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            blend.colorWriteMask = 0xF;
            VkPipelineColorBlendStateCreateInfo blendState{
                VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            blendState.attachmentCount = 1;
            blendState.pAttachments = &blend;
            VkPipelineVertexInputStateCreateInfo emptyInput{
                VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            VkPipelineViewportStateCreateInfo vp{
                VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
            vp.viewportCount = 1;
            vp.scissorCount = 1;
            VkPipelineInputAssemblyStateCreateInfo ia{
                VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            VkPipelineRasterizationStateCreateInfo rs{
                VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            rs.polygonMode = VK_POLYGON_MODE_FILL;
            rs.cullMode = VK_CULL_MODE_NONE;
            rs.lineWidth = 1.0f;
            VkPipelineMultisampleStateCreateInfo ms{
                VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
            ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
            VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            gp.pNext = &rendering;
            gp.stageCount = 2;
            gp.pStages = stages;
            gp.pVertexInputState = &emptyInput;
            gp.pInputAssemblyState = &ia;
            gp.pViewportState = &vp;
            VkPipelineDynamicStateCreateInfo dynamic = dynamicViewportScissor();
            gp.pDynamicState = &dynamic;
            gp.pRasterizationState = &rs;
            gp.pMultisampleState = &ms;
            gp.pColorBlendState = &blendState;
            gp.layout = layout;
            const bool ok = vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp, nullptr, out) ==
                            VK_SUCCESS;
            vkDestroyShaderModule(dev, frag, nullptr);
            if (!ok && error) *error = std::string("failed to create pipeline ") + fs;
            return ok;
        };
        // The bloom pass writes into the 1/4-resolution bloom target; the composite
        // pass writes into whatever the caller presents (swapchain image, usually BGRA8).
        if (!makeFullscreen("bloom.frag.spv", VK_FORMAT_R16G16B16A16_SFLOAT, false,
                            compositePipelineLayout_, &bloomPipeline_))
            return false;
        // Two composite pipelines: one for BGRA8 swapchains, one for RGBA8 offscreen.
        if (!makeFullscreen("composite.frag.spv", VK_FORMAT_B8G8R8A8_UNORM, false,
                            compositePipelineLayout_, &compositePipeline_))
            return false;
        if (!makeFullscreen("composite.frag.spv", VK_FORMAT_R8G8B8A8_UNORM, false,
                            compositePipelineLayout_, &compositePipelineRgba_))
            return false;
        // TAA resolve writes the same RGBA16F HDR format it reads.
        if (!makeFullscreen("taa.frag.spv", VK_FORMAT_R16G16B16A16_SFLOAT, false,
                            taaPipelineLayout_, &taaPipeline_))
            return false;
        vkDestroyShaderModule(dev, vert, nullptr);
    }
    return true;
}

// =============================================================================
//  Upload
// =============================================================================
VkDescriptorSet HairGpuRenderer::allocateModelDescriptors(GpuHairModel* model) {
    VkDevice dev = ctx_->device();
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = descriptorPool_;
    // Bound at indices 1 (model uniforms), 2 (bundle table) and 4 (the five textures);
    // index 0 is the shared frame set and index 3 the shared sample/output set.
    VkDescriptorSetLayout layouts[3] = {modelLayout_, bundleLayout_, hairTextureLayout_};
    ai.descriptorSetCount = 3;
    ai.pSetLayouts = layouts;
    VkDescriptorSet sets[3] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
    if (vkAllocateDescriptorSets(dev, &ai, sets) != VK_SUCCESS) return VK_NULL_HANDLE;

    VkDescriptorBufferInfo uniformInfo{model->uniformBuffer, 0, sizeof(HairModelUniforms)};
    VkDescriptorBufferInfo bundleInfo{model->bundleBuffer, 0, VK_WHOLE_SIZE};
    VkDescriptorImageInfo imageInfos[5];
    VkWriteDescriptorSet writes[7]{};
    for (int i = 0; i < 5; ++i) {
        imageInfos[i].sampler = samplerLinear_;
        imageInfos[i].imageView = model->views[i];
        imageInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = sets[2];
        writes[i].dstBinding = static_cast<uint32_t>(i);
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[i].pImageInfo = &imageInfos[i];
    }
    writes[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[5].dstSet = sets[0];
    writes[5].dstBinding = 0;
    writes[5].descriptorCount = 1;
    writes[5].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[5].pBufferInfo = &uniformInfo;
    writes[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[6].dstSet = sets[1];
    writes[6].dstBinding = 0;
    writes[6].descriptorCount = 1;
    writes[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[6].pBufferInfo = &bundleInfo;
    vkUpdateDescriptorSets(dev, 7, writes, 0, nullptr);
    model->descriptorSets[0] = sets[0];
    model->descriptorSets[1] = sets[1];
    model->descriptorSets[2] = sets[2];
    return sets[2];
}

void HairGpuRenderer::uploadTextureImage(GpuHairModel* model, int index, const float* data,
                                         uint32_t w, uint32_t h, uint32_t d,
                                         uint32_t components) {
    // The textures are stored as R16 float formats (13-21 KB per model, as in Figure 1).
    VkFormat format = components == 1 ? VK_FORMAT_R16_SFLOAT
                     : components == 2 ? VK_FORMAT_R16G16_SFLOAT
                                       : VK_FORMAT_R16G16B16A16_SFLOAT;
    const VkDeviceSize texels = static_cast<VkDeviceSize>(w) * h * d;
    const VkDeviceSize bytes = texels * 4 * sizeof(uint16_t);
    std::vector<uint16_t> packed(static_cast<size_t>(texels) * 4, 0);
    for (VkDeviceSize i = 0; i < texels; ++i) {
        for (uint32_t c = 0; c < components; ++c)
            packed[i * 4 + c] = toHalf(data[i * components + c]);
        if (components < 4) packed[i * 4 + 3] = toHalf(1.0f);
    }

    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    VkBuffer staging = ctx_->createBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                          &stagingMemory);
    void* mapped = nullptr;
    vkMapMemory(ctx_->device(), stagingMemory, 0, bytes, 0, &mapped);
    std::memcpy(mapped, packed.data(), static_cast<size_t>(bytes));
    vkUnmapMemory(ctx_->device(), stagingMemory);

    if (d > 1) {
        model->images[index] = ctx_->createImage3D(
            w, h, d, format,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, &model->memories[index],
            &model->views[index]);
    } else {
        model->images[index] = ctx_->createImage2D(
            w, h, format, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT, &model->memories[index], &model->views[index]);
    }
    VkCommandBuffer cmd = ctx_->beginSingleTimeCommands();
    ctx_->imageLayoutTransition(cmd, model->images[index], VK_IMAGE_LAYOUT_UNDEFINED,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                VK_IMAGE_ASPECT_COLOR_BIT);
    ctx_->copyBufferToImage3D(cmd, staging, model->images[index], w, h, d);
    ctx_->imageLayoutTransition(cmd, model->images[index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                VK_IMAGE_ASPECT_COLOR_BIT);
    ctx_->endSingleTimeCommands(cmd);
    ctx_->destroyBuffer(staging, stagingMemory);

    model->layouts[index] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    model->dims[index][0] = w;
    model->dims[index][1] = h;
    model->dims[index][2] = d;
}

bool HairGpuRenderer::uploadBundleBuffer(GpuHairModel* model, const std::vector<Vec3>& bundleCenter,
                                         const std::vector<float>& bundleSize,
                                         const std::vector<float>& bundleLength,
                                         const std::vector<uint32_t>& strandCounts,
                                         uint32_t seed) {
    const size_t n = strandCounts.size();
    std::vector<float> data(n * 8, 0.0f);  // vec4 bundles[] + vec4 meta[]
    for (size_t i = 0; i < n; ++i) {
        data[i * 4 + 0] = bundleSize[i];                 // h for Eq. 7 (strand LOD)
        data[i * 4 + 1] = static_cast<float>(strandCounts[i]);
        data[i * 4 + 2] = bundleLength[i];               // h for Eq. 7 (vertex LOD)
        data[i * 4 + 3] = 0.0f;
        // isTriangle flag: quad-dominant presets only; triangle bundles are handled by
        // the same code path with the triangle sample table (Section 3.3).
        data[n * 4 + i * 4 + 0] = 0.0f;
        data[n * 4 + i * 4 + 1] = static_cast<float>(strandCounts[i]);
        data[n * 4 + i * 4 + 2] = static_cast<float>(seed);
        data[n * 4 + i * 4 + 3] = 0.0f;
    }
    // The two arrays live in separate descriptors in the shaders; upload them as one
    // buffer with a fixed split at n*16 bytes (see below).
    const VkDeviceSize bytes = data.size() * sizeof(float);
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    VkBuffer staging = ctx_->createBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                          &stagingMemory);
    void* mapped = nullptr;
    vkMapMemory(ctx_->device(), stagingMemory, 0, bytes, 0, &mapped);
    std::memcpy(mapped, data.data(), static_cast<size_t>(bytes));
    vkUnmapMemory(ctx_->device(), stagingMemory);
    model->bundleBuffer = ctx_->createBuffer(bytes,
                                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                             &model->bundleMemory);
    VkCommandBuffer cmd = ctx_->beginSingleTimeCommands();
    VkBufferCopy copy{0, 0, bytes};
    vkCmdCopyBuffer(cmd, staging, model->bundleBuffer, 1, &copy);
    ctx_->endSingleTimeCommands(cmd);
    ctx_->destroyBuffer(staging, stagingMemory);
    model->numFaces = static_cast<uint32_t>(n);
    return model->bundleBuffer != VK_NULL_HANDLE;
}

bool HairGpuRenderer::uploadHairModel(const HairInstance& instance) {
    GpuHairModel model;
    if (!uploadBundleBuffer(&model, instance.bundleCenter, instance.bundleSize,
                            instance.bundleLength, instance.mesh.bundleStrandCounts(),
                            instance.seed))
        return false;

    const HairMeshTextureSet& tex = instance.textures;
    uploadTextureImage(&model, 0, tex.hairMeshTexture().data.data(), tex.width(), tex.height(),
                       tex.depth(), 3);
    uploadTextureImage(&model, 1, tex.uvTexture().data.data(), tex.uvTexture().width,
                       tex.uvTexture().height, 1, 2);
    uploadTextureImage(&model, 2, tex.wTexture().data.data(), tex.width(), tex.height(),
                       tex.wTexture().depth, 1);
    uploadTextureImage(&model, 3, tex.uTexture().data.data(), tex.width(), tex.height(),
                       tex.uTexture().depth, 3);
    uploadTextureImage(&model, 4, tex.vTexture().data.data(), tex.width(), tex.height(),
                       tex.vTexture().depth, 3);

    model.textureBytes = tex.sizeReport(TexelFormat::Float16).total();
    model.maxStrandsPerBundle = 1;
    for (uint32_t c : instance.mesh.bundleStrandCounts()) model.maxStrandsPerBundle = std::max(c, model.maxStrandsPerBundle);

    model.uniformBuffer = ctx_->createBuffer(sizeof(HairModelUniforms),
                                             VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                             &model.uniformMemory);
    if (model.uniformBuffer == VK_NULL_HANDLE) return false;
    vkMapMemory(ctx_->device(), model.uniformMemory, 0, sizeof(HairModelUniforms), 0,
                &model.uniformMapped);

    // Compute-path work item count for this model (Option A of Section 3.2).
    model.chunksPerBundle = (model.maxStrandsPerBundle + 31u) / 32u;
    model.instance = &instance;
    if (!allocateModelDescriptors(&model)) return false;
    models_.push_back(model);
    stats_.textureBytes += model.textureBytes;
    return true;
}

bool HairGpuRenderer::updateDynamicTextures(const HairInstance& instance) {
    for (auto& model : models_) {
        // Table 1: only the hair mesh texture, u-texture and v-texture change.
        const HairMeshTextureSet& tex = instance.textures;
        uploadTextureImage(&model, 0, tex.hairMeshTexture().data.data(), tex.width(), tex.height(),
                           tex.depth(), 3);
        uploadTextureImage(&model, 3, tex.uTexture().data.data(), tex.width(), tex.height(),
                           tex.uTexture().depth, 3);
        uploadTextureImage(&model, 4, tex.vTexture().data.data(), tex.width(), tex.height(),
                           tex.vTexture().depth, 3);
        return true;
    }
    return false;
}

void HairGpuRenderer::writeModelUniforms(GpuHairModel* model, const HairInstance& instance,
                                         const RenderOptions& options, const Camera& camera,
                                         uint32_t height) {
    HairModelUniforms u{};
    copyMat(instance.transform, u.model);
    const Mat4 viewProj = camera.proj() * camera.view();
    copyMat(viewProj * instance.transform, u.modelViewProj);
    copyMat(instance.transform.inverseAffine(), u.modelInv);
    // Light matrix for the shadow pass (orthographic, Eq. 7's orthographic variant).
    const Vec3 sunDir = scene_ ? scene_->sunDirection : normalize(Vec3(0.35f, 0.75f, 0.45f));
    const float sceneRadius = scene_ ? 1.6f : 0.6f;
    const Mat4 lightVP = Mat4::ortho(-sceneRadius, sceneRadius, -sceneRadius, sceneRadius, 0.01f,
                                     sceneRadius * 4.0f) *
                         Mat4::lookAt(normalize(sunDir) * (sceneRadius * 2.0f), Vec3(0, 0, 0),
                                      Vec3(0, 1, 0));
    copyMat(lightVP * instance.transform, u.lightModelViewProj);

    const HairMeshTextureSet& tex = instance.textures;
    u.textureSize[0] = static_cast<float>(tex.width());
    u.textureSize[1] = static_cast<float>(tex.height());
    u.textureSize[2] = static_cast<float>(tex.depth());
    u.textureSize[3] = static_cast<float>(tex.numLayers());
    u.blocks[0] = static_cast<float>(tex.layout().blocksX);
    u.blocks[1] = static_cast<float>(tex.layout().blocksY);
    u.blocks[2] = static_cast<float>(tex.layout().gutter);
    u.blocks[3] = static_cast<float>(tex.numFaces());
    u.cameraPos[0] = camera.position.x;
    u.cameraPos[1] = camera.position.y;
    u.cameraPos[2] = camera.position.z;
    u.cameraPos[3] = 1.0f;
    u.viewportSize[0] = static_cast<float>(width_);
    u.viewportSize[1] = static_cast<float>(height_);
    u.viewportSize[2] = 1.0f / static_cast<float>(width_);
    u.viewportSize[3] = 1.0f / static_cast<float>(height_);
    u.lodParams[0] = options.lod.alphaStrand;
    u.lodParams[1] = options.lod.alphaVertex;
    u.lodParams[2] = options.lod.lambda;
    u.lodParams[3] = options.lodBias;
    u.lodScreen[0] = static_cast<float>(height);
    u.lodScreen[1] = std::tan(camera.fovY);
    const StylingParams& sp = instance.styling;
    u.styling0[0] = sp.clumping;
    u.styling0[1] = sp.clumpScale;
    u.styling0[2] = sp.clumpTipTightness;
    u.styling0[3] = sp.curl;
    u.styling1[0] = sp.curlFrequency;
    u.styling1[1] = sp.curlRadius;
    u.styling1[2] = sp.frizz;
    u.styling1[3] = sp.frizzFrequency;
    u.styling4[0] = sp.clumpRadiusM;
    u.styling4[1] = sp.curlRadiusM;
    u.styling4[2] = 0.0f;
    u.styling4[3] = 0.0f;
    u.styling2[0] = sp.straightness;
    u.styling2[1] = sp.rootFade;
    u.styling2[2] = sp.lengthJitter;
    u.styling2[3] = sp.volume;
    u.styling3[0] = sp.randomOffset;
    u.styling3[1] = instance.baseRadius;
    u.styling3[2] = sp.amplitudeScale;
    u.styling3[3] = static_cast<float>(instance.seed);
    u.hairColor[0] = sp.color.x;
    u.hairColor[1] = sp.color.y;
    u.hairColor[2] = sp.color.z;
    u.hairColor[3] = sp.melanin;
    u.hairParams[0] = static_cast<float>(instance.seed);
    u.hairParams[1] = static_cast<float>(model->maxStrandsPerBundle);
    u.hairParams[2] = static_cast<float>(model->maxStrandsPerBundle);
    u.hairParams[3] = 0.0f;  // gamma is supplied per bundle through the task payload
    std::memcpy(model->uniformMapped, &u, sizeof(HairModelUniforms));
}

// =============================================================================
//  Frame recording
// =============================================================================
bool HairGpuRenderer::prepareComputeWorkItems(const Camera& cam, const RenderOptions& opt,
                                            bool uploadGeometry) {
    // "Option A" of Section 3.2: the CPU pre-pass performs the task-shader decision
    // (strand counts from Eq. 7/8 and the vertex LOD exponent) and fills the work-item
    // table consumed by the compute shader. This runs once per frame for the whole scene.
    std::vector<float> items;
    uint32_t totalIndices = 0;
    uint64_t strandsRendered = 0, strandsTotal = 0, triangles = 0;
    int minE = 9, maxE = -1;
    for (auto& model : models_) {
        const HairInstance& inst = *model.instance;
        const auto& counts = inst.mesh.bundleStrandCounts();
        for (uint32_t f = 0; f < counts.size(); ++f) {
            strandsTotal += counts[f];
            const Vec3 center = inst.transform.transformPoint(inst.bundleCenter[f]);
            const float dist = lodDistance(cam.position, center, inst.bundleSize[f] * 0.5f);
            const float L = opt.enableLod
                                ? detailLevel(inst.bundleSize[f], static_cast<float>(height_), dist * opt.lodBias,
                                              cam.fovY, opt.lod.alphaStrand)
                                : 1.0f;
            const uint32_t n = opt.enableLod
                                   ? strandCountLod(counts[f], bundleDelta(f, inst.seed), L)
                                   : counts[f];
            float strandLength = 0.0f;
            {
                Vec3 lo, hi;
                inst.mesh.bounds(&lo, &hi);
                strandLength = std::max(hi.y - lo.y, 1e-3f);
            }
            const float Lv = opt.enableLod
                                 ? detailLevel(strandLength, static_cast<float>(height_), dist * opt.lodBias,
                                               cam.fovY, opt.lod.alphaVertex)
                                 : 1.0f;
            const VertexLodResult vl = computeVertexLod(Lv, 32, opt.lod.lambda);
            const float thickness = opt.enableLod ? strandThicknessScale(L, counts[f]) : 1.0f;
            minE = std::min(minE, vl.exponentE);
            maxE = std::max(maxE, vl.exponentE);
            for (uint32_t s = 0; s < n; ++s) {
                items.push_back(static_cast<float>(f));
                items.push_back(static_cast<float>(s));
                items.push_back(static_cast<float>(vl.exponentE));
                items.push_back(thickness);
                totalIndices += 6u * (1u << vl.exponentE);
            }
            strandsRendered += n;
            triangles += 2ull * (1ull << vl.exponentE) * n;
        }
    }
    const uint32_t workItems = static_cast<uint32_t>(items.size() / 4);
    stats_.strandsRendered = strandsRendered;
    stats_.strandsTotal = strandsTotal;
    stats_.triangles = triangles;
    stats_.vertexLodMin = minE == 9 ? 0 : minE;
    stats_.vertexLodMax = std::max(maxE, 0);
    stats_.taskShaderCount = 0;
    for (const auto& model : models_)
        stats_.taskShaderCount += static_cast<int>(model.numFaces * model.chunksPerBundle);
    stats_.meshShaderRuns = static_cast<int>(workItems);
    stats_.crowdCount = static_cast<int>(models_.size());
    computeWorkItems_ = workItems;
    computeIndexCount_ = totalIndices;
    computeMaxSegments_ = maxE < 0 ? 1u : (1u << std::min<uint32_t>(maxE, 5u));
    if (!uploadGeometry) return true;   // the task/mesh path reads no work items

    // The compute path needs one uniform ribbon stride for the whole dispatch, so the
    // buffers are grown to 2*ns_max+2 vertices per work item when the LOD changes.
    const uint32_t strideVerts = 2u * computeMaxSegments_ + 2u;
    const VkDeviceSize needVerts =
        VkDeviceSize(workItems) * strideVerts * sizeof(float) * 4u;
    if (workItems > maxStrands_ || needVerts > computeVertexCapacity_) {
        const uint32_t capacity = std::max(workItems, 1u) + std::max(workItems / 4u, 1024u);
        if (!createComputeFallbackBuffers(capacity)) {
            stats_.warning = "failed to grow the compute-path buffers";
            return false;
        }
        computeVertexCapacity_ = VkDeviceSize(capacity) * strideVerts * sizeof(float) * 4u;
        writeSampleDescriptors();
    }

    // Upload the work item table.
    const VkDeviceSize bytes = items.size() * sizeof(float);
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    VkBuffer staging = ctx_->createBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                          &stagingMemory);
    void* mapped = nullptr;
    vkMapMemory(ctx_->device(), stagingMemory, 0, bytes, 0, &mapped);
    std::memcpy(mapped, items.data(), static_cast<size_t>(bytes));
    vkUnmapMemory(ctx_->device(), stagingMemory);
    VkCommandBuffer cmd = ctx_->beginSingleTimeCommands();
    VkBufferCopy copy{0, 0, bytes};
    vkCmdCopyBuffer(cmd, staging, workItemBuffer_, 1, &copy);
    ctx_->endSingleTimeCommands(cmd);
    ctx_->destroyBuffer(staging, stagingMemory);

    computeIndexCount_ = totalIndices;
    return true;
}

namespace {

// The per-model push constants both the task/mesh path and the compute fallback consume.
HairPush makeHairPush(const GpuHairModel& model, const RenderOptions& opt, bool shadowPass) {
    HairPush push{};
    push.chunksPerBundle = model.chunksPerBundle;
    push.tasksPerChunk = 32;
    push.frameIndex = 0;
    push.flags = shadowPass ? 1u : 0u;
    const StylingParams& sp = model.instance->styling;
    push.shadeColor[0] = sp.color.x;
    push.shadeColor[1] = sp.color.y;
    push.shadeColor[2] = sp.color.z;
    push.shadeColor[3] = sp.melanin;
    push.shadeParams[0] = sp.melanin;
    push.shadeParams[1] = sp.redness;
    push.shadeParams[2] = opt.enableLod ? 0.6f : 0.0f;   // dual scattering for thick LOD strands
    push.shadeParams[3] = 0.0f;
    push.shadeParams2[0] = opt.hairAlpha;
    push.shadeParams2[1] = opt.drawShadows ? 1.0f : 0.0f;
    push.shadeParams2[2] = 1.0f;
    push.shadeParams2[3] = 0.5f;
    return push;
}

}  // namespace

// Section 3.2, "Option A": the compute fallback generates every ribbon with one dispatch
// per hair model. Dispatches cannot be recorded inside a render pass instance, so this runs
// once per frame before the shadow and colour passes, and both of them then rasterise the
// same storage-buffer output with the vertex-path pipelines.
void HairGpuRenderer::recordComputeFallback(VkCommandBuffer cmd, const RenderOptions& opt) {
    if (hasMeshShaders_ || hairComputePipeline_ == VK_NULL_HANDLE) return;
    if (computeWorkItems_ == 0) return;
    bool dispatched = false;
    for (auto& model : models_) {
        if (model.descriptorSets[0] == VK_NULL_HANDLE || !model.instance) continue;
        const VkDescriptorSet sets[5] = {frameDescriptorSet_, model.descriptorSets[0],
                                         model.descriptorSets[1], sampleDescriptorSet_,
                                         model.descriptorSets[2]};
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hairPipelineLayout_, 0, 5,
                                sets, 0, nullptr);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hairComputePipeline_);
        HairPush push = makeHairPush(model, opt, false);
        vkCmdPushConstants(cmd, hairPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(HairPush), &push);
        vkCmdDispatch(cmd, (computeWorkItems_ + 31) / 32, 1, 1);
        dispatched = true;
    }
    if (!dispatched) return;
    // Make the generated ribbons visible to the vertex-input/vertex stages of both passes.
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDEX_READ_BIT |
                            VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                         0, 1, &barrier, 0, nullptr, 0, nullptr);
}

void HairGpuRenderer::recordHairDraw(VkCommandBuffer cmd, const Camera& cam,
                                     const RenderOptions& opt, bool shadowPass) {
    const uint32_t renderW = width_, renderH = height_;
    VkViewport viewport{0.0f, 0.0f, static_cast<float>(renderW), static_cast<float>(renderH),
                        0.0f, 1.0f};
    VkRect2D scissor{{0, 0}, {renderW, renderH}};
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    for (size_t i = 0; i < models_.size(); ++i) {
        GpuHairModel& model = models_[i];
        const HairInstance& inst = *model.instance;
        if (model.descriptorSets[0] == VK_NULL_HANDLE) continue;

        const HairPush push = makeHairPush(model, opt, shadowPass);

        // set 0 frame, 1 model, 2 bundle table, 3 sample/output, 4 the five textures
        const VkDescriptorSet sets[5] = {frameDescriptorSet_,   model.descriptorSets[0],
                                         model.descriptorSets[1], sampleDescriptorSet_,
                                         model.descriptorSets[2]};
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, hairPipelineLayout_, 0, 5,
                                sets, 0, nullptr);
        if (hasMeshShaders_ && hairMeshPipeline_ != VK_NULL_HANDLE) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              shadowPass ? shadowHairPipeline_ : hairMeshPipeline_);
            vkCmdPushConstants(cmd, hairPipelineLayout_,
                               VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT |
                                   VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(HairPush), &push);
            cmdDrawMeshTasksExt_(cmd, model.numFaces * model.chunksPerBundle, 1, 1);
        } else {
            // The ribbons were generated by recordComputeFallback() before either pass; the
            // vertex path only has to rasterise them, exactly as the mesh path does.
            if (computeIndexCount_ > 0) {
                VkPipeline drawPipeline = shadowPass ? hairShadowDrawPipeline_ : hairDrawPipeline_;
                if (drawPipeline != VK_NULL_HANDLE) {
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, drawPipeline);
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                            drawPipelineLayout_, 0, 5, sets, 0, nullptr);
                    struct DrawPush {
                        float mvp[16];
                        float model[16];
                        float params[4];
                        float params2[4];
                    } dp{};
                    // The shadow pass needs the light's orthographic matrix instead.
                    const Mat4 frameViewProj =
                        shadowPass ? lightViewProj_ : cam.proj() * cam.view();
                    copyMat(frameViewProj * inst.transform, dp.mvp);
                    copyMat(inst.transform, dp.model);
                    vkCmdPushConstants(cmd, drawPipelineLayout_,
                                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                       0, sizeof(dp), &dp);
                    vkCmdBindIndexBuffer(cmd, strandIndexBuffer_, 0, VK_INDEX_TYPE_UINT32);
                    vkCmdDrawIndexed(cmd, computeIndexCount_, 1, 0, 0, 0);
                }
            }
        }
    }
}

void HairGpuRenderer::recordShadowPass(VkCommandBuffer cmd, const Camera& cam) {
    // Shadow map: depth-only pass, orthographic light view (Eq. 7's orthographic variant).
    const uint32_t res = 2048;
    VkViewport viewport{0.0f, 0.0f, static_cast<float>(res), static_cast<float>(res), 0.0f, 1.0f};
    VkRect2D scissor{{0, 0}, {res, res}};
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    RenderOptions shadowOpt;
    shadowOpt.enableLod = true;
    shadowOpt.drawShadows = false;
    recordHairDraw(cmd, cam, shadowOpt, true);
}

void HairGpuRenderer::recordBloomPass(VkCommandBuffer cmd, const RenderOptions& opt) {
    const uint32_t bw = std::max(width_ / 4, 1u), bh = std::max(height_ / 4, 1u);
    ctx_->imageLayoutTransition(cmd, bloomImage_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                VK_IMAGE_ASPECT_COLOR_BIT);
    VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    color.imageView = bloomView_;
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.clearValue.color = {{0.0f, 0.0f, 0.0f, 0.0f}};
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, {bw, bh}};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &color;
    vkCmdBeginRendering(cmd, &ri);
    VkViewport vp{0.0f, 0.0f, static_cast<float>(bw), static_cast<float>(bh), 0.0f, 1.0f};
    VkRect2D sc{{0, 0}, {bw, bh}};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, bloomPipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, compositePipelineLayout_, 0, 1,
                            &bloomSet_, 0, nullptr);
    float params[4] = {0.9f, 1.0f / static_cast<float>(bw), 0.0f, 0.0f};
    vkCmdPushConstants(cmd, compositePipelineLayout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(params), params);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRendering(cmd);
    ctx_->imageLayoutTransition(cmd, bloomImage_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                VK_IMAGE_ASPECT_COLOR_BIT);
    (void)opt;
}

void HairGpuRenderer::setCompositeTarget(VkImageView targetView, VkFormat targetFormat) {
    compositeTargetView_ = targetView;
    compositeTargetFormat_ = targetFormat;
}

void HairGpuRenderer::recordComposite(VkCommandBuffer cmd, VkImage target, uint32_t width,
                                      uint32_t height, const RenderOptions& opt) {
    ctx_->imageLayoutTransition(cmd, target, VK_IMAGE_LAYOUT_UNDEFINED,
                                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                VK_IMAGE_ASPECT_COLOR_BIT);
    VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    // The composite pass writes into the externally owned image; the caller provides the
    // matching view (QVulkanWindow's swapchain view, or our own offscreen view).
    color.imageView = compositeTargetView_;
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.clearValue.color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, {width, height}};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &color;
    vkCmdBeginRendering(cmd, &ri);
    VkViewport vp{0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f};
    VkRect2D sc{{0, 0}, {width, height}};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      compositeTargetFormat_ == VK_FORMAT_B8G8R8A8_UNORM ? compositePipeline_
                                                                          : compositePipelineRgba_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, compositePipelineLayout_, 0, 1,
                            &compositeSet_, 0, nullptr);
    float params[4] = {opt.exposure, 0.06f, 0.0f, 1.0f};
    vkCmdPushConstants(cmd, compositePipelineLayout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(params), params);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRendering(cmd);
}

bool HairGpuRenderer::renderFrame(const Camera& camera, const RenderOptions& options,
                                  const std::string& screenshotPath) {
    using clock = std::chrono::high_resolution_clock;
    const auto start = clock::now();
    if (!ctx_) return false;

    if (width_ != static_cast<uint32_t>(options.width) ||
        height_ != static_cast<uint32_t>(options.height))
        resize(static_cast<uint32_t>(options.width), static_cast<uint32_t>(options.height));

    // MSAA: changing the toggle rebuilds the main-pass pipelines and the multisampled
    // attachments; unsupported counts are clamped to what the device can resolve.
    if (options.msaa != msaaSamples_ && !setSampleCount(options.msaa))
        stats_.warning = "MSAA " + std::to_string(options.msaa) + "x is not supported here; " +
                         "rendering at " + std::to_string(msaaSamples_) + "x";

    // ---- temporal AA: advance the sub-pixel jitter, then use one jittered projection for
    // the whole frame (geometry, light matrices, resolve).
    taaEnabled_ = options.taa;
    if (taaEnabled_ && (resolvedImage_ == VK_NULL_HANDLE || resolvedView_ == VK_NULL_HANDLE ||
                        width_ != resolvedWidth_ || height_ != resolvedHeight_)) {
        createTaaResources();
        taaResetRequested_ = true;
    }
    if (taaEnabled_) advanceJitter();
    const Mat4 proj = jitteredProj(camera);
    const Mat4 viewProj = proj * camera.view();
    copyMat(viewProj, currentViewProj_);
    copyMat(viewProj.inverseAffine(), currentInvViewProj_);
    if (taaEnabled_) bindPresentationImages(taaHistoryValid_);

    // Uniforms.
    CameraUniforms camU{};
    copyMat(camera.view(), camU.view);
    copyMat(proj, camU.proj);
    camU.cameraPos[0] = camera.position.x;
    camU.cameraPos[1] = camera.position.y;
    camU.cameraPos[2] = camera.position.z;
    camU.cameraPos[3] = 1.0f;
    camU.sunDirection[0] = scene_ ? scene_->sunDirection.x : 0.35f;
    camU.sunDirection[1] = scene_ ? scene_->sunDirection.y : 0.75f;
    camU.sunDirection[2] = scene_ ? scene_->sunDirection.z : 0.45f;
    camU.sunDirection[3] = 2.2f;
    camU.sunColor[0] = 1.0f; camU.sunColor[1] = 0.93f; camU.sunColor[2] = 0.82f; camU.sunColor[3] = 0.0f;
    camU.skyZenith[0] = 0.20f; camU.skyZenith[1] = 0.36f; camU.skyZenith[2] = 0.62f; camU.skyZenith[3] = 0.0f;
    camU.skyHorizon[0] = 0.62f; camU.skyHorizon[1] = 0.70f; camU.skyHorizon[2] = 0.82f; camU.skyHorizon[3] = 0.0f;
    camU.lodScreen[0] = static_cast<float>(height_);
    // Eq. 7's frustum height per unit distance: 2 tan(theta/2), with theta the vertical FOV.
    camU.lodScreen[1] = 2.0f * std::tan(0.5f * camera.fovY);
    std::memcpy(cameraUniformMapped_, &camU, sizeof(camU));

    LightUniforms light{};
    const Vec3 sunDir = scene_ ? scene_->sunDirection : normalize(Vec3(0.35f, 0.75f, 0.45f));
    const float sceneRadius = scene_ ? 1.6f : 0.6f;
    const Mat4 lightVP = Mat4::ortho(-sceneRadius, sceneRadius, -sceneRadius, sceneRadius, 0.01f,
                                     sceneRadius * 4.0f) *
                         Mat4::lookAt(normalize(sunDir) * (sceneRadius * 2.0f), Vec3(0, 0, 0),
                                      Vec3(0, 1, 0));
    copyMat(lightVP, light.lightViewProj);
    lightViewProj_ = lightVP;
    light.shadowParams[0] = 1.0f / 2048.0f;
    light.shadowParams[1] = 0.0015f;
    light.shadowParams[2] = 1.5f;
    light.shadowParams[3] = options.drawShadows ? 1.0f : 0.0f;
    std::memcpy(lightUniformMapped_, &light, sizeof(light));

    for (auto& model : models_)
        writeModelUniforms(&model, *model.instance, options, camera, height_);

    // Composite uniforms (used by the sky/grid reconstruction and tone mapping).
    {
        CompositeUniforms cu{};
        copyMat((camera.proj() * camera.view()).inverseAffine(), cu.invViewProj);
        cu.cameraPos[0] = camera.position.x;
        cu.cameraPos[1] = camera.position.y;
        cu.cameraPos[2] = camera.position.z;
        cu.cameraPos[3] = 1.0f;
        const Vec3 sun = scene_ ? scene_->sunDirection : normalize(Vec3(0.35f, 0.75f, 0.45f));
        cu.sunDirection[0] = sun.x; cu.sunDirection[1] = sun.y; cu.sunDirection[2] = sun.z;
        cu.sunDirection[3] = 1.0f;
        cu.sunColor[0] = 1.0f; cu.sunColor[1] = 0.93f; cu.sunColor[2] = 0.82f; cu.sunColor[3] = 0.0f;
        cu.skyZenith[0] = 0.20f; cu.skyZenith[1] = 0.36f; cu.skyZenith[2] = 0.62f;
        cu.skyHorizon[0] = 0.62f; cu.skyHorizon[1] = 0.70f; cu.skyHorizon[2] = 0.82f;
        cu.groundColor[0] = 0.16f; cu.groundColor[1] = 0.15f; cu.groundColor[2] = 0.14f;
        cu.params[0] = options.exposure;
        cu.params[1] = 0.06f;
        cu.params[2] = 0.0f;
        cu.params[3] = (options.drawSky || options.drawGround) ? 1.0f : 0.0f;
        cu.viewport[0] = static_cast<float>(width_);
        cu.viewport[1] = static_cast<float>(height_);
        cu.viewport[2] = 1.0f / static_cast<float>(width_);
        cu.viewport[3] = 1.0f / static_cast<float>(height_);
        std::memcpy(compositeUniformMapped_, &cu, sizeof(cu));
    }

    // Work-item table (Option A) and the shared statistics for both paths.
    const bool useCompute = !hasMeshShaders_;
    if (!prepareComputeWorkItems(camera, options, useCompute)) return false;

    if (!ctx_ || commandBuffer_ == VK_NULL_HANDLE) return false;
    vkWaitForFences(ctx_->device(), 1, &frameFence_, VK_TRUE, UINT64_MAX);
    vkResetFences(ctx_->device(), 1, &frameFence_);
    vkResetCommandBuffer(commandBuffer_, 0);

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBuffer_, &bi);

    // ---- compute fallback: generate the ribbons once, outside any render pass instance
    // (a compute dispatch may not be recorded inside one); the shadow pass and the colour
    // pass then rasterise the same buffers. ----
    if (options.drawHair) recordComputeFallback(commandBuffer_, options);

    // ---- shadow pass ----
    {
        ctx_->imageLayoutTransition(commandBuffer_, shadowImage_,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                    VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                    VK_IMAGE_ASPECT_DEPTH_BIT);
        VkRenderingAttachmentInfo depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        depth.imageView = shadowView_;
        depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depth.clearValue.depthStencil = {1.0f, 0};
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
        ri.renderArea = {{0, 0}, {2048, 2048}};
        ri.layerCount = 1;
        ri.pDepthAttachment = &depth;
        vkCmdBeginRendering(commandBuffer_, &ri);
        if (options.drawShadows && options.drawHair) recordShadowPass(commandBuffer_, camera);
        vkCmdEndRendering(commandBuffer_);
        ctx_->imageLayoutTransition(commandBuffer_, shadowImage_,
                                    VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                    VK_IMAGE_ASPECT_DEPTH_BIT);
    }

    // ---- main HDR pass (multisampled when MSAA is on, resolved into hdrImage_) ----
    ctx_->imageLayoutTransition(commandBuffer_, hdrImage_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                VK_IMAGE_ASPECT_COLOR_BIT);
    VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    color.imageView = hdrView_;
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.clearValue.color = {{0.0f, 0.0f, 0.0f, 0.0f}};
    VkRenderingAttachmentInfo depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    depth.imageView = depthView_;
    depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depth.clearValue.depthStencil = {1.0f, 0};

    const bool multisampled = msaaSamples_ > 1 && msaaView_ != VK_NULL_HANDLE &&
                              msaaDepthView_ != VK_NULL_HANDLE;
    if (multisampled) {
        // Contents of the MSAA targets are always fully cleared, so UNDEFINED is the
        // cheapest valid source layout (the same idiom the depth image uses).
        ctx_->imageLayoutTransition(commandBuffer_, msaaImage_, VK_IMAGE_LAYOUT_UNDEFINED,
                                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_IMAGE_ASPECT_COLOR_BIT);
        ctx_->imageLayoutTransition(commandBuffer_, msaaDepthImage_, VK_IMAGE_LAYOUT_UNDEFINED,
                                    VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                    VK_IMAGE_ASPECT_DEPTH_BIT);
        color.imageView = msaaView_;
        color.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
        color.resolveImageView = hdrView_;
        color.resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        depth.imageView = msaaDepthView_;
    } else {
        ctx_->imageLayoutTransition(commandBuffer_, depthImage_, VK_IMAGE_LAYOUT_UNDEFINED,
                                    VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                    VK_IMAGE_ASPECT_DEPTH_BIT);
    }
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, {width_, height_}};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &color;
    ri.pDepthAttachment = &depth;
    vkCmdBeginRendering(commandBuffer_, &ri);

    VkViewport vp{0.0f, 0.0f, static_cast<float>(width_), static_cast<float>(height_), 0.0f, 1.0f};
    VkRect2D sc{{0, 0}, {width_, height_}};
    vkCmdSetViewport(commandBuffer_, 0, 1, &vp);
    vkCmdSetScissor(commandBuffer_, 0, 1, &sc);
    const VkDeviceSize zeroOffsets[1] = {0};

    if (options.drawHead && headPipeline_ != VK_NULL_HANDLE && scene_) {
        vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_GRAPHICS, headPipeline_);
        const Mat4 headViewProj = proj * camera.view();
        auto drawMesh = [&](const StaticMeshGpu& mesh, VkBuffer indexBuffer, Vec3 albedo,
                            float spec) {
            if (mesh.indexCount == 0) return;
            HeadPush push{};
            copyMat(headViewProj, push.mvp);
            copyMat(Mat4::identity(), push.model);
            push.albedo[0] = albedo.x;
            push.albedo[1] = albedo.y;
            push.albedo[2] = albedo.z;
            push.albedo[3] = spec;
            vkCmdPushConstants(commandBuffer_, headPipelineLayout_,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                               sizeof(HeadPush), &push);
            vkCmdBindVertexBuffers(commandBuffer_, 0, 1, &mesh.buffer, zeroOffsets);
            vkCmdBindIndexBuffer(commandBuffer_, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(commandBuffer_, mesh.indexCount, 1, 0, 0, 0);
        };
        // A crowd draws the shared head/body mesh once per hair instance, with that
        // instance's transform; a single character draws it with the identity transform.
        auto drawMeshXf = [&](const StaticMeshGpu& mesh, VkBuffer indexBuffer, Vec3 albedo,
                              float spec, const Mat4& xform) {
            if (mesh.indexCount == 0) return;
            HeadPush push{};
            copyMat(headViewProj * xform, push.mvp);
            copyMat(xform, push.model);
            push.albedo[0] = albedo.x;
            push.albedo[1] = albedo.y;
            push.albedo[2] = albedo.z;
            push.albedo[3] = spec;
            vkCmdPushConstants(commandBuffer_, headPipelineLayout_,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                               sizeof(HeadPush), &push);
            vkCmdBindVertexBuffers(commandBuffer_, 0, 1, &mesh.buffer, zeroOffsets);
            vkCmdBindIndexBuffer(commandBuffer_, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(commandBuffer_, mesh.indexCount, 1, 0, 0, 0);
        };
        if (models_.size() > 1) {
            for (const auto& model : models_) {
                if (!model.instance) continue;
                drawMeshXf(headMesh_, headMeshIndex_.buffer, Vec3(0.655f, 0.45f, 0.37f), -1.0f,
                           model.instance->transform);
                drawMeshXf(eyesMesh_, eyesMeshIndex_.buffer, Vec3(0.16f, 0.11f, 0.07f), -2.0f,
                           model.instance->transform);
                drawMeshXf(bodyMesh_, bodyMeshIndex_.buffer, Vec3(0.635f, 0.435f, 0.36f), 0.0f,
                           model.instance->transform);
            }
        } else {
            drawMesh(headMesh_, headMeshIndex_.buffer, Vec3(0.655f, 0.45f, 0.37f), -1.0f);
            drawMesh(eyesMesh_, eyesMeshIndex_.buffer, Vec3(0.16f, 0.11f, 0.07f), -2.0f);
            drawMesh(bodyMesh_, bodyMeshIndex_.buffer, Vec3(0.635f, 0.435f, 0.36f), 0.0f);
        }
        // Figure 9: the teapot under each hair layer.
        if (teapotMesh_.indexCount > 0) {
            if (models_.size() > 1) {
                for (const auto& model : models_)
                    if (model.instance)
                        drawMeshXf(teapotMesh_, teapotMeshIndex_.buffer,
                                   Vec3(0.86f, 0.84f, 0.80f), 0.0f, model.instance->transform);
            } else {
                drawMesh(teapotMesh_, teapotMeshIndex_.buffer, Vec3(0.86f, 0.84f, 0.80f), 0.0f);
            }
        }
    }

    if (options.drawHair) recordHairDraw(commandBuffer_, camera, options, false);
    vkCmdEndRendering(commandBuffer_);

    // ---- temporal resolve, then bloom over the resolved image ----
    if (taaEnabled_ && taaPipeline_ != VK_NULL_HANDLE) {
        ctx_->imageLayoutTransition(commandBuffer_, hdrImage_,
                                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                    VK_IMAGE_ASPECT_COLOR_BIT);
        recordTaaPass(commandBuffer_, taaResetRequested_);
        taaResetRequested_ = false;
        // Bloom and the composite now sample the resolved image.
        bindPresentationImages(taaHistoryValid_ || resolvedView_ != VK_NULL_HANDLE);
    } else {
        ctx_->imageLayoutTransition(commandBuffer_, hdrImage_,
                                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                    VK_IMAGE_ASPECT_COLOR_BIT);
    }
    if (bloomPipeline_ != VK_NULL_HANDLE) recordBloomPass(commandBuffer_, options);

    // ---- tone mapping / sky / grid: composite into the LDR target ----
    // The windowed path calls renderToSwapchainImage(), which records the same composite
    // pass into the swapchain image instead.
    if (!screenshotPath.empty() && compositePipelineRgba_ != VK_NULL_HANDLE) {
        setCompositeTarget(ldrView_, VK_FORMAT_R8G8B8A8_UNORM);
        recordComposite(commandBuffer_, ldrImage_, width_, height_, options);
        ctx_->imageLayoutTransition(commandBuffer_, ldrImage_,
                                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                    VK_IMAGE_ASPECT_COLOR_BIT);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {width_, height_, 1};
        vkCmdCopyImageToBuffer(commandBuffer_, ldrImage_,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readbackBuffer_, 1, &region);
        ctx_->imageLayoutTransition(commandBuffer_, ldrImage_,
                                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_IMAGE_ASPECT_COLOR_BIT);
        compositeTargetView_ = VK_NULL_HANDLE;
    }

    vkEndCommandBuffer(commandBuffer_);
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &commandBuffer_;
    SILK_VK_CHECK(vkQueueSubmit(ctx_->graphicsQueue(), 1, &submit, frameFence_));
    vkWaitForFences(ctx_->device(), 1, &frameFence_, VK_TRUE, UINT64_MAX);

    if (!screenshotPath.empty()) readbackToPng(screenshotPath);

    ++frameCounter_;
    std::memcpy(prevViewProj_, currentViewProj_, sizeof(prevViewProj_));

    stats_.frameMs = std::chrono::duration<double, std::milli>(clock::now() - start).count();
    stats_.crowdCount = static_cast<int>(models_.size());
    stats_.msaa = msaaSamples_;
    stats_.meshShaderPath = hasMeshShaders_;
    stats_.taa = taaEnabled_;
    return true;
}

bool HairGpuRenderer::readbackToPng(const std::string& path) {
    if (!readbackMapped_) return false;
    // The readback buffer holds the tone-mapped R8G8B8A8 image produced by the composite
    // pass; convert it to the shared Image type and write a PNG with the dependency-free
    // encoder used by the software path.
    Image img;
    img.resize(static_cast<int>(width_), static_cast<int>(height_));
    const uint8_t* rgba = static_cast<const uint8_t*>(readbackMapped_);
    for (uint32_t y = 0; y < height_; ++y)
        for (uint32_t x = 0; x < width_; ++x) {
            const size_t idx = (static_cast<size_t>(y) * width_ + x) * 4;
            img.at(static_cast<int>(x), static_cast<int>(y)) =
                Vec3(rgba[idx + 0] / 255.0f, rgba[idx + 1] / 255.0f, rgba[idx + 2] / 255.0f);
        }
    return writePng(path, img, 1.0f);
}

bool HairGpuRenderer::renderToSwapchainImage(VkImage image, uint32_t width, uint32_t height,
                                             const Camera& camera, const RenderOptions& options) {
    // Runs the full offscreen frame (shadow pass, hair pass, bloom) and then composites
    // into the externally owned image, leaving it in PRESENT_SRC_KHR.
    if (!renderFrame(camera, options, std::string())) return false;
    vkResetFences(ctx_->device(), 1, &frameFence_);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkResetCommandBuffer(commandBuffer_, 0);
    vkBeginCommandBuffer(commandBuffer_, &bi);
    recordComposite(commandBuffer_, image, width, height, options);
    ctx_->imageLayoutTransition(commandBuffer_, image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_ASPECT_COLOR_BIT);
    vkEndCommandBuffer(commandBuffer_);
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &commandBuffer_;
    if (vkQueueSubmit(ctx_->graphicsQueue(), 1, &submit, frameFence_) != VK_SUCCESS) return false;
    vkWaitForFences(ctx_->device(), 1, &frameFence_, VK_TRUE, UINT64_MAX);
    return true;
}

void HairGpuRenderer::resize(uint32_t width, uint32_t height) {
    if (!ctx_ || width == 0 || height == 0) return;
    if (width == width_ && height == height_ && hdrImage_ != VK_NULL_HANDLE) return;
    width_ = width;
    height_ = height;
    if (hdrImage_ == VK_NULL_HANDLE) {
        createRenderTargets();
        return;
    }
    vkDeviceWaitIdle(ctx_->device());
    createRenderTargets();   // destroys and recreates every resolution-dependent target
    createTaaResources();
    taaResetRequested_ = true;
    // The composite descriptor set samples hdr/bloom, so it is rewritten as well.
    if (compositeSet_ != VK_NULL_HANDLE) {
        VkDescriptorImageInfo imgs[2] = {
            {samplerLinear_, hdrView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            {samplerLinear_, bloomView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}};
        VkWriteDescriptorSet writes[2]{};
        for (int i = 0; i < 2; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = compositeSet_;
            writes[i].dstBinding = static_cast<uint32_t>(i);
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[i].pImageInfo = &imgs[i];
        }
        vkUpdateDescriptorSets(ctx_->device(), 2, writes, 0, nullptr);
        VkDescriptorImageInfo src{samplerLinear_, hdrView_,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet bw{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        bw.dstSet = bloomSet_;
        bw.dstBinding = 0;
        bw.descriptorCount = 1;
        bw.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bw.pImageInfo = &src;
        vkUpdateDescriptorSets(ctx_->device(), 1, &bw, 0, nullptr);
    }
}

void HairGpuRenderer::setScene(std::shared_ptr<Scene> scene) {
    scene_ = std::move(scene);
    if (!scene_) return;
    stats_.textureBytes = 0;
    if (ctx_ && ctx_->device() != VK_NULL_HANDLE && !models_.empty()) {
        vkDeviceWaitIdle(ctx_->device());
        for (auto& model : models_) {
            for (int i = 0; i < 5; ++i)
                if (model.images[i] != VK_NULL_HANDLE)
                    ctx_->destroyImage(model.images[i], model.memories[i], model.views[i]);
            if (model.bundleBuffer != VK_NULL_HANDLE)
                ctx_->destroyBuffer(model.bundleBuffer, model.bundleMemory);
            if (model.uniformBuffer != VK_NULL_HANDLE)
                ctx_->destroyBuffer(model.uniformBuffer, model.uniformMemory);
        }
    }
    models_.clear();
    for (const auto& inst : scene_->hair) uploadHairModel(*inst);
    uploadStaticMeshes();
    uint32_t totalStrands = 0;
    for (const auto& inst : scene_->hair)
        for (uint32_t c : inst->mesh.bundleStrandCounts()) totalStrands += c;
    stats_.strandsTotal = totalStrands;
}

void HairGpuRenderer::applyHardwarePolicy(const DeviceInfo& info, RenderOptions* options) const {
    if (!options) return;
    options->msaa = info.defaultMsaa;
    options->lodBias = info.lodBias;
    options->enableLod = true;
    if (info.tier == HardwareTier::Software) {
        options->hairAlpha = 0.5f;
        options->ssaa = 1;
    }
}

}  // namespace silk

namespace silk {

// =============================================================================
//  Static geometry upload (head, body, teapot)
// =============================================================================
namespace {

// Interleaved layout used by head.vert: position(3) normal(3) uv(2).
std::vector<float> interleaveMesh(const Mesh& mesh) {
    std::vector<float> data(mesh.positions.size() * 8);
    for (size_t i = 0; i < mesh.positions.size(); ++i) {
        const Vec3& p = mesh.positions[i];
        const Vec3 n = mesh.normals.empty() ? Vec3(0, 1, 0) : mesh.normals[i];
        const Vec2 t = mesh.uvs.empty() ? Vec2(0, 0) : mesh.uvs[i];
        data[i * 8 + 0] = p.x; data[i * 8 + 1] = p.y; data[i * 8 + 2] = p.z;
        data[i * 8 + 3] = n.x; data[i * 8 + 4] = n.y; data[i * 8 + 5] = n.z;
        data[i * 8 + 6] = t.x; data[i * 8 + 7] = t.y;
    }
    return data;
}

}  // namespace

bool HairGpuRenderer::uploadStaticMeshes() {
    if (!scene_) return false;
    auto upload = [&](const Mesh& mesh, StaticMeshGpu* vertexOut, StaticMeshGpu* indexOut) {
        if (mesh.indices.empty()) return;
        const std::vector<float> verts = interleaveMesh(mesh);
        const VkDeviceSize vb = verts.size() * sizeof(float);
        const VkDeviceSize ib = mesh.indices.size() * sizeof(uint32_t);

        VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
        VkBuffer staging = ctx_->createBuffer(vb + ib, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                              &stagingMemory);
        void* mapped = nullptr;
        vkMapMemory(ctx_->device(), stagingMemory, 0, vb + ib, 0, &mapped);
        std::memcpy(mapped, verts.data(), static_cast<size_t>(vb));
        std::memcpy(static_cast<char*>(mapped) + vb, mesh.indices.data(),
                    static_cast<size_t>(ib));
        vkUnmapMemory(ctx_->device(), stagingMemory);

        vertexOut->buffer = ctx_->createBuffer(vb, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                                                       VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                               &vertexOut->memory);
        indexOut->buffer = ctx_->createBuffer(ib, VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                              &indexOut->memory);
        VkCommandBuffer cmd = ctx_->beginSingleTimeCommands();
        VkBufferCopy vcopy{0, 0, vb};
        VkBufferCopy icopy{vb, 0, ib};
        vkCmdCopyBuffer(cmd, staging, vertexOut->buffer, 1, &vcopy);
        vkCmdCopyBuffer(cmd, staging, indexOut->buffer, 1, &icopy);
        ctx_->endSingleTimeCommands(cmd);
        ctx_->destroyBuffer(staging, stagingMemory);
        vertexOut->indexCount = static_cast<uint32_t>(mesh.indices.size());
        indexOut->indexCount = static_cast<uint32_t>(mesh.indices.size());
    };
    upload(scene_->head, &headMesh_, &headMeshIndex_);
    upload(scene_->eyes, &eyesMesh_, &eyesMeshIndex_);
    upload(scene_->body, &bodyMesh_, &bodyMeshIndex_);
    upload(scene_->teapot, &teapotMesh_, &teapotMeshIndex_);
    return true;
}

}  // namespace silk

namespace silk {

// =============================================================================
//  Descriptor set layouts
// =============================================================================
bool HairGpuRenderer::createDescriptorLayouts(std::string* error) {
    VkDevice dev = ctx_->device();
    const VkShaderStageFlags allStages = VK_SHADER_STAGE_ALL;

    auto make = [&](const std::vector<VkDescriptorSetLayoutBinding>& bindings,
                    VkDescriptorSetLayout* out, const char* what) -> bool {
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        ci.bindingCount = static_cast<uint32_t>(bindings.size());
        ci.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(dev, &ci, nullptr, out) != VK_SUCCESS) {
            if (error) *error = std::string("failed to create the ") + what + " layout";
            return false;
        }
        return true;
    };

    // set 0: per-frame data (camera, shadow map, light). Shared by every pipeline.
    {
        VkDescriptorSetLayoutBinding b[3]{};
        b[0].binding = 0;
        b[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        b[0].descriptorCount = 1;
        b[0].stageFlags = allStages;
        b[1].binding = 1;
        b[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[1].descriptorCount = 1;
        b[1].stageFlags = allStages;
        b[2].binding = 2;
        b[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        b[2].descriptorCount = 1;
        b[2].stageFlags = allStages;
        if (!make({b[0], b[1], b[2]}, &frameLayout_, "frame")) return false;
    }
    // set 1: per-hair-model uniforms (no instancing of hair data).
    {
        VkDescriptorSetLayoutBinding b{};
        b.binding = 0;
        b.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        b.descriptorCount = 1;
        b.stageFlags = allStages;
        if (!make({b}, &modelLayout_, "model")) return false;
    }
    // set 2: per-hair-model bundle table (one vec4 per bundle: center + size/length).
    {
        VkDescriptorSetLayoutBinding b{};
        b.binding = 0;
        b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b.descriptorCount = 1;
        b.stageFlags = allStages;
        if (!make({b}, &bundleLayout_, "bundle")) return false;
    }
    // set 3: shared root sample tables and compute-path output buffers.
    {
        std::vector<VkDescriptorSetLayoutBinding> b(7);
        for (int i = 0; i < 7; ++i) {
            b[i].binding = static_cast<uint32_t>(i);
            b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            b[i].descriptorCount = 1;
            b[i].stageFlags = allStages;
        }
        if (!make(b, &sampleLayout_, "sample/output")) return false;
    }
    // set 4: the five textures of Table 1.
    {
        std::vector<VkDescriptorSetLayoutBinding> b(5);
        for (int i = 0; i < 5; ++i) {
            b[i].binding = static_cast<uint32_t>(i);
            b[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b[i].descriptorCount = 1;
            b[i].stageFlags = allStages;
        }
        if (!make(b, &hairTextureLayout_, "hair texture")) return false;
    }

    // set 5: post-process composite (HDR colour + bloom + uniforms).
    {
        VkDescriptorSetLayoutBinding b[3]{};
        for (int i = 0; i < 2; ++i) {
            b[i].binding = static_cast<uint32_t>(i);
            b[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        b[2].binding = 2;
        b[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        b[2].descriptorCount = 1;
        b[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        if (!make({b[0], b[1], b[2]}, &compositeLayout_, "composite")) return false;
    }
    // set 6: bloom (single input image).
    {
        VkDescriptorSetLayoutBinding b{};
        b.binding = 0;
        b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        if (!make({b}, &bloomLayout_, "bloom")) return false;
    }
    // set 7: TAA resolve (current colour, history, uniforms).
    {
        VkDescriptorSetLayoutBinding b[3]{};
        for (int i = 0; i < 2; ++i) {
            b[i].binding = static_cast<uint32_t>(i);
            b[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        b[2].binding = 2;
        b[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        b[2].descriptorCount = 1;
        b[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        if (!make({b[0], b[1], b[2]}, &taaLayout_, "taa")) return false;
    }

    // The composite/bloom/TAA layouts used to be created inside createPipelines(), after the
    // descriptor sets that use them had already been allocated - so those sets were allocated
    // against VK_NULL_HANDLE layouts. Validation reported it and lavapipe segfaulted later, in
    // vkCreateGraphicsPipelines. Every layout now lives here, and every one is checked.
    const std::pair<const char*, VkDescriptorSetLayout> required[] = {
        {"frame", frameLayout_},         {"model", modelLayout_},
        {"bundle", bundleLayout_},       {"sample/output", sampleLayout_},
        {"hair texture", hairTextureLayout_}, {"composite", compositeLayout_},
        {"bloom", bloomLayout_},         {"taa", taaLayout_}};
    for (const auto& layout : required) {
        if (layout.second == VK_NULL_HANDLE) {
            if (error) *error = std::string("descriptor layout missing: ") + layout.first;
            return false;
        }
    }
    return true;
}

// =============================================================================
//  Render targets
// =============================================================================
// =============================================================================
//  MSAA (the paper renders with 8x MSAA; the toggle is exposed in the CLI and the
//  editor). Only the main HDR pass is multisampled: the shadow map, bloom chain,
//  TAA resolve and composite stay single-sampled, and hdrImage_ receives the resolve.
// =============================================================================
bool HairGpuRenderer::createMsaaTargets() {
    if (msaaImage_ != VK_NULL_HANDLE) ctx_->destroyImage(msaaImage_, msaaMemory_, msaaView_);
    if (msaaDepthImage_ != VK_NULL_HANDLE)
        ctx_->destroyImage(msaaDepthImage_, msaaDepthMemory_, msaaDepthView_);
    msaaImage_ = VK_NULL_HANDLE;
    msaaMemory_ = VK_NULL_HANDLE;
    msaaView_ = VK_NULL_HANDLE;
    msaaDepthImage_ = VK_NULL_HANDLE;
    msaaDepthMemory_ = VK_NULL_HANDLE;
    msaaDepthView_ = VK_NULL_HANDLE;
    if (msaaSamples_ <= 1 || !ctx_) return true;

    const VkSampleCountFlagBits samples = sampleCountFlag(msaaSamples_);
    msaaImage_ = ctx_->createImage2D(width_, height_, VK_FORMAT_R16G16B16A16_SFLOAT,
                                     VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                         VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                     VK_IMAGE_ASPECT_COLOR_BIT, &msaaMemory_, &msaaView_, samples);
    if (msaaImage_ == VK_NULL_HANDLE) return false;
    msaaDepthImage_ = ctx_->createImage2D(width_, height_, VK_FORMAT_D32_SFLOAT,
                                          VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                                          VK_IMAGE_ASPECT_DEPTH_BIT, &msaaDepthMemory_,
                                          &msaaDepthView_, samples);
    return msaaDepthImage_ != VK_NULL_HANDLE;
}

bool HairGpuRenderer::setSampleCount(int wanted) {
    if (!ctx_) return false;
    if (wanted < 1) wanted = 1;

    // Clamp to what this device can actually resolve for colour and depth.
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(ctx_->physicalDevice(), &props);
    int chosen = 1;
    for (const int candidate : {8, 4, 2}) {
        if (candidate > wanted) continue;
        const VkSampleCountFlagBits bits = sampleCountFlag(candidate);
        if ((props.limits.framebufferColorSampleCounts & bits) &&
            (props.limits.framebufferDepthSampleCounts & bits)) {
            chosen = candidate;
            break;
        }
    }
    if (chosen == msaaSamples_) return chosen == wanted;

    vkDeviceWaitIdle(ctx_->device());
    destroyMainPassPipelines();
    const int previous = msaaSamples_;
    msaaSamples_ = chosen;
    if (!createMsaaTargets()) {
        msaaSamples_ = 1;
        createMsaaTargets();
    }
    std::string error;
    if (!createMainPassPipelines(&error)) {
        stats_.warning = "could not build the " + std::to_string(msaaSamples_) +
                         "x MSAA pipelines (" + error + "); falling back to 1x";
        destroyMainPassPipelines();
        msaaSamples_ = 1;
        createMsaaTargets();
        createMainPassPipelines(nullptr);
        return false;
    }
    (void)previous;
    return msaaSamples_ == wanted;
}

bool HairGpuRenderer::createRenderTargets() {
    auto destroyIfAny = [&](VkImage& img, VkDeviceMemory& mem, VkImageView& view) {
        if (img != VK_NULL_HANDLE) ctx_->destroyImage(img, mem, view);
        img = VK_NULL_HANDLE;
        mem = VK_NULL_HANDLE;
        view = VK_NULL_HANDLE;
    };
    destroyIfAny(hdrImage_, hdrMemory_, hdrView_);
    destroyIfAny(depthImage_, depthMemory_, depthView_);
    destroyIfAny(bloomImage_, bloomMemory_, bloomView_);
    destroyIfAny(shadowImage_, shadowMemory_, shadowView_);
    destroyIfAny(ldrImage_, ldrMemory_, ldrView_);
    if (!createMsaaTargets()) return false;

    hdrImage_ = ctx_->createImage2D(width_, height_, VK_FORMAT_R16G16B16A16_SFLOAT,
                                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                        VK_IMAGE_USAGE_SAMPLED_BIT |
                                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                    VK_IMAGE_ASPECT_COLOR_BIT, &hdrMemory_, &hdrView_);
    if (hdrImage_ == VK_NULL_HANDLE) return false;
    depthImage_ = ctx_->createImage2D(width_, height_, VK_FORMAT_D32_SFLOAT,
                                      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                          VK_IMAGE_USAGE_SAMPLED_BIT,
                                      VK_IMAGE_ASPECT_DEPTH_BIT, &depthMemory_, &depthView_);
    if (depthImage_ == VK_NULL_HANDLE) return false;
    bloomImage_ = ctx_->createImage2D(std::max(width_ / 4u, 1u), std::max(height_ / 4u, 1u),
                                      VK_FORMAT_R16G16B16A16_SFLOAT,
                                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                          VK_IMAGE_USAGE_SAMPLED_BIT,
                                      VK_IMAGE_ASPECT_COLOR_BIT, &bloomMemory_, &bloomView_);
    if (bloomImage_ == VK_NULL_HANDLE) return false;
    ldrImage_ = ctx_->createImage2D(width_, height_, VK_FORMAT_R8G8B8A8_UNORM,
                                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                    VK_IMAGE_ASPECT_COLOR_BIT, &ldrMemory_, &ldrView_);
    if (ldrImage_ == VK_NULL_HANDLE) return false;
    shadowImage_ = ctx_->createImage2D(2048, 2048, VK_FORMAT_D32_SFLOAT,
                                       VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                           VK_IMAGE_USAGE_SAMPLED_BIT,
                                       VK_IMAGE_ASPECT_DEPTH_BIT, &shadowMemory_, &shadowView_);
    if (shadowImage_ == VK_NULL_HANDLE) return false;

    // Bring every target into the layout the frame loop expects to transition *from*.
    VkCommandBuffer cmd = ctx_->beginSingleTimeCommands();
    ctx_->imageLayoutTransition(cmd, hdrImage_, VK_IMAGE_LAYOUT_UNDEFINED,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                VK_IMAGE_ASPECT_COLOR_BIT);
    ctx_->imageLayoutTransition(cmd, bloomImage_, VK_IMAGE_LAYOUT_UNDEFINED,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                VK_IMAGE_ASPECT_COLOR_BIT);
    ctx_->imageLayoutTransition(cmd, shadowImage_, VK_IMAGE_LAYOUT_UNDEFINED,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                VK_IMAGE_ASPECT_DEPTH_BIT);
    ctx_->endSingleTimeCommands(cmd);

    // ---- host-visible readback buffer for --screenshot (RGBA8) ----
    if (readbackBuffer_ != VK_NULL_HANDLE) ctx_->destroyBuffer(readbackBuffer_, readbackMemory_);
    const VkDeviceSize readbackBytes = VkDeviceSize(width_) * height_ * 4u;
    readbackBuffer_ = ctx_->createBuffer(readbackBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                         &readbackMemory_);
    if (readbackBuffer_ == VK_NULL_HANDLE) return false;
    vkMapMemory(ctx_->device(), readbackMemory_, 0, readbackBytes, 0, &readbackMapped_);
    return true;
}

// =============================================================================
//  Compute-path buffers ("Option A" of Section 3.2)
// =============================================================================
bool HairGpuRenderer::createComputeFallbackBuffers(uint32_t maxStrands) {
    if (maxStrands == 0) maxStrands = 1;
    const uint32_t segs = computeMaxSegments_;
    const VkDeviceSize vertexBytes = VkDeviceSize(maxStrands) * (2u * segs + 2u) * sizeof(float) * 4u;
    const VkDeviceSize indexBytes = VkDeviceSize(maxStrands) * (2u * segs * 3u) * sizeof(uint32_t);
    const VkDeviceSize metaBytes = VkDeviceSize(maxStrands) * 16u;
    const VkDeviceSize workBytes = VkDeviceSize(maxStrands) * 16u;

    auto recreate = [&](VkBuffer& buffer, VkDeviceMemory& memory, VkDeviceSize size,
                        VkBufferUsageFlags usage) -> bool {
        if (buffer != VK_NULL_HANDLE) ctx_->destroyBuffer(buffer, memory);
        buffer = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
        buffer = ctx_->createBuffer(size, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &memory);
        return buffer != VK_NULL_HANDLE;
    };
    if (!recreate(strandVertexBuffer_, strandVertexMemory_, vertexBytes,
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT))
        return false;
    if (!recreate(strandTangentBuffer_, strandTangentMemory_, vertexBytes,
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT))
        return false;
    if (!recreate(strandIndexBuffer_, strandIndexMemory_, indexBytes,
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT))
        return false;
    if (!recreate(strandMetaBuffer_, strandMetaMemory_, metaBytes,
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT))
        return false;
    if (!recreate(workItemBuffer_, workItemMemory_, workBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT))
        return false;
    maxStrands_ = maxStrands;
    computeVertexCapacity_ = vertexBytes;
    return true;
}

// =============================================================================
//  Shader loading
// =============================================================================
VkShaderModule HairGpuRenderer::loadShader(const char* path, std::string* error) const {
    std::ifstream in(std::string(SILKCROWD_SHADER_DIR) + "/" + path, std::ios::binary);
    if (!in) in.open(std::string("shaders/") + path, std::ios::binary);
    if (!in) {
        if (error) *error = std::string("cannot open shader ") + path;
        return VK_NULL_HANDLE;
    }
    const std::vector<char> code((std::istreambuf_iterator<char>(in)),
                                 std::istreambuf_iterator<char>());
    if (code.empty() || (code.size() % 4) != 0) {
        if (error) *error = std::string("shader is not valid SPIR-V: ") + path;
        return VK_NULL_HANDLE;
    }
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = code.size();
    ci.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(ctx_->device(), &ci, nullptr, &module) != VK_SUCCESS) {
        if (error) *error = std::string("vkCreateShaderModule failed for ") + path;
        return VK_NULL_HANDLE;
    }
    return module;
}

// =============================================================================
//  Lifetime
// =============================================================================
bool HairGpuRenderer::init(VulkanContext* context, uint32_t width, uint32_t height,
                           bool enableValidation, std::string* error) {
    if (!context || context->device() == VK_NULL_HANDLE) {
        if (error) *error = "VulkanContext is not initialised";
        return false;
    }
    ctx_ = context;
    width_ = std::max(width, 1u);
    height_ = std::max(height, 1u);
    validation_ = enableValidation;
    // Mesh shaders need the extension entry point, not just the extension: resolve it the
    // portable way (see the member declaration) and fall back to the compute path if the
    // loader cannot provide it.
    cmdDrawMeshTasksExt_ = reinterpret_cast<PFN_vkCmdDrawMeshTasksEXT>(
        vkGetDeviceProcAddr(ctx_->device(), "vkCmdDrawMeshTasksEXT"));
    hasMeshShaders_ = ctx_->info().meshShaders && !forceCompute_ && cmdDrawMeshTasksExt_ != nullptr;
    computeMaxSegments_ = 8;   // 2^3 segments: the compute path's default vertex budget

    const DeviceInfo& info = ctx_->info();
    stats_ = GpuStats{};
    stats_.deviceName = info.name;
    stats_.vulkanVersion = info.apiVersion;
    stats_.meshShaderPath = hasMeshShaders_;
    stats_.msaa = 1;

    VkDevice dev = ctx_->device();

    // ---- descriptor pool ----
    {
        // Every hair model owns 3 sets; 8 more are shared (frame, samples, composite,
        // bloom and a little slack for re-uploads).
        VkDescriptorPoolSize sizes[3]{};
        sizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sizes[0].descriptorCount = 8 * 5 + 32;
        sizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        sizes[1].descriptorCount = 8 * 2 + 32;
        sizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        sizes[2].descriptorCount = 8 * 2 + 32;
        VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        ci.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT |
                   VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        ci.maxSets = 64;
        ci.poolSizeCount = 3;
        ci.pPoolSizes = sizes;
        SILK_VK_CHECK(vkCreateDescriptorPool(dev, &ci, nullptr, &descriptorPool_));
    }

    // ---- samplers ----
    {
        VkSamplerCreateInfo ci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        ci.magFilter = VK_FILTER_LINEAR;
        ci.minFilter = VK_FILTER_LINEAR;
        ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ci.maxLod = 0.0f;
        SILK_VK_CHECK(vkCreateSampler(dev, &ci, nullptr, &samplerLinear_));
        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = VK_FILTER_LINEAR;
        sci.minFilter = VK_FILTER_LINEAR;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.compareEnable = VK_TRUE;
        sci.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        SILK_VK_CHECK(vkCreateSampler(dev, &sci, nullptr, &samplerShadow_));
    }

    // ---- per-frame uniform buffers ----
    cameraUniform_ = ctx_->createBuffer(sizeof(CameraUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                        &cameraUniformMemory_);
    if (cameraUniform_ == VK_NULL_HANDLE) return false;
    vkMapMemory(dev, cameraUniformMemory_, 0, sizeof(CameraUniforms), 0, &cameraUniformMapped_);
    lightUniform_ = ctx_->createBuffer(sizeof(LightUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                       &lightUniformMemory_);
    if (lightUniform_ == VK_NULL_HANDLE) return false;
    vkMapMemory(dev, lightUniformMemory_, 0, sizeof(LightUniforms), 0, &lightUniformMapped_);

    // ---- progressive blue-noise root sample tables (Yuksel 2015, Section 3.3) ----
    {
        SampleEliminationParams qp;
        qp.targetCount = 2048;
        qp.triangleDomain = false;
        const SampleSet2D quads = generateBlueNoiseSamples(qp);
        SampleEliminationParams tp;
        tp.targetCount = 2048;
        tp.triangleDomain = true;
        tp.seed = 987654321u;
        const SampleSet2D tris = generateBlueNoiseSamples(tp);

        auto upload = [&](const std::vector<Vec2>& pts, VkBuffer* buffer, VkDeviceMemory* memory) {
            const VkDeviceSize bytes = VkDeviceSize(pts.size()) * sizeof(Vec2);
            VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
            VkBuffer staging = ctx_->createBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                                  &stagingMemory);
            void* mapped = nullptr;
            vkMapMemory(dev, stagingMemory, 0, bytes, 0, &mapped);
            std::memcpy(mapped, pts.data(), static_cast<size_t>(bytes));
            vkUnmapMemory(dev, stagingMemory);
            *buffer = ctx_->createBuffer(bytes,
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memory);
            VkCommandBuffer cmd = ctx_->beginSingleTimeCommands();
            VkBufferCopy copy{0, 0, bytes};
            vkCmdCopyBuffer(cmd, staging, *buffer, 1, &copy);
            ctx_->endSingleTimeCommands(cmd);
            ctx_->destroyBuffer(staging, stagingMemory);
            return *buffer != VK_NULL_HANDLE;
        };
        if (!upload(quads.samples, &sampleBuffer_, &sampleMemory_)) return false;
        if (!upload(tris.samples, &sampleTriangleBuffer_, &sampleTriangleMemory_)) return false;
    }

    if (!createRenderTargets()) return false;
    if (!createDescriptorLayouts(error)) return false;

    // ---- compute-path output buffers ("Option A") ----
    if (!createComputeFallbackBuffers(1u << 16)) {
        if (error) *error = "failed to allocate the compute fallback buffers";
        return false;
    }

    // ---- frame descriptor set (set 0) ----
    {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = descriptorPool_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &frameLayout_;
        SILK_VK_CHECK(vkAllocateDescriptorSets(dev, &ai, &frameDescriptorSet_));
        VkDescriptorBufferInfo cam{cameraUniform_, 0, sizeof(CameraUniforms)};
        VkDescriptorBufferInfo lit{lightUniform_, 0, sizeof(LightUniforms)};
        VkDescriptorImageInfo shadow{samplerShadow_, shadowView_,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet writes[3]{};
        for (int i = 0; i < 3; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = frameDescriptorSet_;
            writes[i].dstBinding = static_cast<uint32_t>(i);
            writes[i].descriptorCount = 1;
        }
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].pBufferInfo = &cam;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].pImageInfo = &shadow;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[2].pBufferInfo = &lit;
        vkUpdateDescriptorSets(dev, 3, writes, 0, nullptr);
    }

    // ---- command buffer + fence ----
    {
        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = ctx_->graphicsFamily();
        SILK_VK_CHECK(vkCreateCommandPool(dev, &pci, nullptr, &commandPool_));
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = commandPool_;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        SILK_VK_CHECK(vkAllocateCommandBuffers(dev, &ai, &commandBuffer_));
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        SILK_VK_CHECK(vkCreateFence(dev, &fci, nullptr, &frameFence_));
    }

    if (!createPipelines(error)) return false;

    // Software fallback message: the compute path has no task/mesh shader counts.
    if (!hasMeshShaders_)
        stats_.warning =
            "mesh shaders unavailable: using the compute-shader fallback (Section 3.2, Option A)";
    return true;
}

void HairGpuRenderer::shutdown() {
    if (!ctx_ || ctx_->device() == VK_NULL_HANDLE) return;
    VkDevice dev = ctx_->device();
    vkDeviceWaitIdle(dev);

    for (auto& model : models_) {
        for (int i = 0; i < 5; ++i)
            if (model.images[i] != VK_NULL_HANDLE)
                ctx_->destroyImage(model.images[i], model.memories[i], model.views[i]);
        if (model.bundleBuffer != VK_NULL_HANDLE)
            ctx_->destroyBuffer(model.bundleBuffer, model.bundleMemory);
        if (model.uniformBuffer != VK_NULL_HANDLE)
            ctx_->destroyBuffer(model.uniformBuffer, model.uniformMemory);
    }
    models_.clear();

    VkPipeline pipelines[] = {hairMeshPipeline_, hairComputePipeline_, hairDrawPipeline_,
                              shadowHairPipeline_,   headPipeline_,       compositePipeline_,
                              compositePipelineRgba_, bloomPipeline_};
    for (VkPipeline p : pipelines)
        if (p != VK_NULL_HANDLE) vkDestroyPipeline(dev, p, nullptr);
    hairMeshPipeline_ = hairComputePipeline_ = hairDrawPipeline_ = shadowHairPipeline_ =
        headPipeline_ = compositePipeline_ = compositePipelineRgba_ = bloomPipeline_ =
            VK_NULL_HANDLE;

    VkPipelineLayout layouts[] = {hairPipelineLayout_, headPipelineLayout_,
                                  compositePipelineLayout_, drawPipelineLayout_};
    for (VkPipelineLayout l : layouts)
        if (l != VK_NULL_HANDLE) vkDestroyPipelineLayout(dev, l, nullptr);
    hairPipelineLayout_ = headPipelineLayout_ = compositePipelineLayout_ = drawPipelineLayout_ =
        VK_NULL_HANDLE;

    VkDescriptorSetLayout setLayouts[] = {hairTextureLayout_, modelLayout_, bundleLayout_,
                                          sampleLayout_,      frameLayout_, compositeLayout_,
                                          bloomLayout_};
    for (VkDescriptorSetLayout l : setLayouts)
        if (l != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(dev, l, nullptr);
    hairTextureLayout_ = modelLayout_ = bundleLayout_ = sampleLayout_ = frameLayout_ =
        compositeLayout_ = bloomLayout_ = VK_NULL_HANDLE;

    if (descriptorPool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(dev, descriptorPool_, nullptr);
    descriptorPool_ = VK_NULL_HANDLE;

    ctx_->destroyImage(msaaImage_, msaaMemory_, msaaView_);
    ctx_->destroyImage(msaaDepthImage_, msaaDepthMemory_, msaaDepthView_);
    ctx_->destroyImage(hdrImage_, hdrMemory_, hdrView_);
    ctx_->destroyImage(bloomImage_, bloomMemory_, bloomView_);
    ctx_->destroyImage(depthImage_, depthMemory_, depthView_);
    ctx_->destroyImage(shadowImage_, shadowMemory_, shadowView_);
    ctx_->destroyImage(resolvedImage_, resolvedMemory_, resolvedView_);
    ctx_->destroyImage(historyImage_, historyMemory_, historyView_);
    hdrImage_ = bloomImage_ = depthImage_ = shadowImage_ = resolvedImage_ = historyImage_ =
        VK_NULL_HANDLE;
    msaaImage_ = msaaDepthImage_ = VK_NULL_HANDLE;
    msaaMemory_ = msaaDepthMemory_ = VK_NULL_HANDLE;
    msaaView_ = msaaDepthView_ = VK_NULL_HANDLE;

    const std::pair<VkBuffer, VkDeviceMemory> owned[] = {
        {sampleBuffer_, sampleMemory_},
        {sampleTriangleBuffer_, sampleTriangleMemory_},
        {strandVertexBuffer_, strandVertexMemory_},
        {strandTangentBuffer_, strandTangentMemory_},
        {strandIndexBuffer_, strandIndexMemory_},
        {strandMetaBuffer_, strandMetaMemory_},
        {workItemBuffer_, workItemMemory_},
        {cameraUniform_, cameraUniformMemory_},
        {lightUniform_, lightUniformMemory_},
        {compositeUniform_, compositeUniformMemory_},
        {headMesh_.buffer, headMesh_.memory},
        {headMeshIndex_.buffer, headMeshIndex_.memory},
        {eyesMesh_.buffer, eyesMesh_.memory},
        {eyesMeshIndex_.buffer, eyesMeshIndex_.memory},
        {bodyMesh_.buffer, bodyMesh_.memory},
        {bodyMeshIndex_.buffer, bodyMeshIndex_.memory},
        {teapotMesh_.buffer, teapotMesh_.memory},
        {teapotMeshIndex_.buffer, teapotMeshIndex_.memory},
    };
    for (const auto& entry : owned)
        if (entry.first != VK_NULL_HANDLE) ctx_->destroyBuffer(entry.first, entry.second);
    sampleBuffer_ = sampleTriangleBuffer_ = strandVertexBuffer_ = strandTangentBuffer_ =
        strandIndexBuffer_ = strandMetaBuffer_ = workItemBuffer_ = cameraUniform_ =
            lightUniform_ = compositeUniform_ = VK_NULL_HANDLE;
    headMesh_ = StaticMeshGpu{};
    eyesMesh_ = StaticMeshGpu{};
    bodyMesh_ = StaticMeshGpu{};
    teapotMesh_ = StaticMeshGpu{};
    headMeshIndex_ = StaticMeshGpu{};
    eyesMeshIndex_ = StaticMeshGpu{};
    bodyMeshIndex_ = StaticMeshGpu{};
    teapotMeshIndex_ = StaticMeshGpu{};

    if (samplerLinear_ != VK_NULL_HANDLE) vkDestroySampler(dev, samplerLinear_, nullptr);
    if (samplerShadow_ != VK_NULL_HANDLE) vkDestroySampler(dev, samplerShadow_, nullptr);
    samplerLinear_ = samplerShadow_ = VK_NULL_HANDLE;

    if (taaUniform_ != VK_NULL_HANDLE) ctx_->destroyBuffer(taaUniform_, taaUniformMemory_);
    taaUniform_ = VK_NULL_HANDLE;
    if (taaLayout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(dev, taaLayout_, nullptr);
    taaLayout_ = VK_NULL_HANDLE;
    if (taaPipelineLayout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(dev, taaPipelineLayout_, nullptr);
    taaPipelineLayout_ = VK_NULL_HANDLE;
    if (taaPipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(dev, taaPipeline_, nullptr);
    taaPipeline_ = VK_NULL_HANDLE;

    if (frameFence_ != VK_NULL_HANDLE) vkDestroyFence(dev, frameFence_, nullptr);
    if (commandPool_ != VK_NULL_HANDLE) vkDestroyCommandPool(dev, commandPool_, nullptr);
    frameFence_ = VK_NULL_HANDLE;
    commandBuffer_ = VK_NULL_HANDLE;
    commandPool_ = VK_NULL_HANDLE;
    ctx_ = nullptr;
}

}  // namespace silk

namespace silk {

// =============================================================================
//  Temporal anti-aliasing
// =============================================================================
bool HairGpuRenderer::createTaaResources() {
    VkDevice dev = ctx_->device();
    auto destroyIfAny = [&](VkImage& img, VkDeviceMemory& mem, VkImageView& view) {
        if (img != VK_NULL_HANDLE) ctx_->destroyImage(img, mem, view);
        img = VK_NULL_HANDLE;
        mem = VK_NULL_HANDLE;
        view = VK_NULL_HANDLE;
    };
    destroyIfAny(resolvedImage_, resolvedMemory_, resolvedView_);
    destroyIfAny(historyImage_, historyMemory_, historyView_);
    const VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                    VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    resolvedImage_ = ctx_->createImage2D(width_, height_, VK_FORMAT_R16G16B16A16_SFLOAT, usage,
                                         VK_IMAGE_ASPECT_COLOR_BIT, &resolvedMemory_,
                                         &resolvedView_);
    if (resolvedImage_ == VK_NULL_HANDLE) return false;
    historyImage_ = ctx_->createImage2D(width_, height_, VK_FORMAT_R16G16B16A16_SFLOAT, usage,
                                        VK_IMAGE_ASPECT_COLOR_BIT, &historyMemory_,
                                        &historyView_);
    if (historyImage_ == VK_NULL_HANDLE) return false;

    VkCommandBuffer cmd = ctx_->beginSingleTimeCommands();
    ctx_->imageLayoutTransition(cmd, resolvedImage_, VK_IMAGE_LAYOUT_UNDEFINED,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                VK_IMAGE_ASPECT_COLOR_BIT);
    ctx_->imageLayoutTransition(cmd, historyImage_, VK_IMAGE_LAYOUT_UNDEFINED,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                VK_IMAGE_ASPECT_COLOR_BIT);
    ctx_->endSingleTimeCommands(cmd);

    if (taaUniform_ == VK_NULL_HANDLE) {
        taaUniform_ = ctx_->createBuffer(sizeof(TaaUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                         &taaUniformMemory_);
        if (taaUniform_ == VK_NULL_HANDLE) return false;
        vkMapMemory(dev, taaUniformMemory_, 0, sizeof(TaaUniforms), 0, &taaUniformMapped_);
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = descriptorPool_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &taaLayout_;
        if (vkAllocateDescriptorSets(dev, &ai, &taaSet_) != VK_SUCCESS) return false;
    }

    VkDescriptorImageInfo cur{samplerLinear_, hdrView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo hist{samplerLinear_, historyView_,
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorBufferInfo ubo{taaUniform_, 0, sizeof(TaaUniforms)};
    VkWriteDescriptorSet writes[3]{};
    for (int i = 0; i < 3; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = taaSet_;
        writes[i].dstBinding = static_cast<uint32_t>(i);
        writes[i].descriptorCount = 1;
    }
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].pImageInfo = &cur;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[1].pImageInfo = &hist;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[2].pBufferInfo = &ubo;
    vkUpdateDescriptorSets(dev, 3, writes, 0, nullptr);
    resolvedWidth_ = width_;
    resolvedHeight_ = height_;
    taaHistoryValid_ = false;
    return true;
}

// Halton(2,3) sub-pixel jitter, cycled over 8 frames. Advanced once per frame, before the
// main pass, so that the projection the geometry is rendered with and the projection the
// resolve reprojects from are the same one.
void HairGpuRenderer::advanceJitter() {
    auto halton = [](uint32_t index, uint32_t base) {
        float f = 1.0f, r = 0.0f;
        uint32_t i = index;
        while (i > 0) {
            f /= static_cast<float>(base);
            r += f * static_cast<float>(i % base);
            i /= base;
        }
        return r;
    };
    const uint32_t n = frameCounter_ + 1;
    prevJitter_[0] = jitter_[0];
    prevJitter_[1] = jitter_[1];
    const float scale = 1.0f;   // jitter magnitude in pixels, before the NDC conversion
    jitter_[0] = (halton((n % 8u) + 1u, 2u) - 0.5f) * scale;
    jitter_[1] = (halton((n % 8u) + 1u, 3u) - 0.5f) * scale;
}

// Clip-space translation by the sub-pixel jitter: M' = T(ndcJitter) * M, so the whole scene
// is rendered half a pixel off and the temporal resolve reconstructs the exact result.
Mat4 HairGpuRenderer::jitteredProj(const Camera& cam) const {
    Mat4 proj = cam.proj();
    if (!taaEnabled_) return proj;
    const float ndcX = 2.0f * jitter_[0] / static_cast<float>(std::max(width_, 1u));
    const float ndcY = 2.0f * jitter_[1] / static_cast<float>(std::max(height_, 1u));
    return Mat4::translate(Vec3(ndcX, ndcY, 0.0f)) * proj;
}

// Points the bloom/composite descriptors at the TAA-resolved HDR image (or the raw HDR
// target when TAA is off).
void HairGpuRenderer::bindPresentationImages(bool useResolved) {
    if (!ctx_ || compositeSet_ == VK_NULL_HANDLE) return;
    VkImageView srcView = useResolved && resolvedView_ != VK_NULL_HANDLE ? resolvedView_ : hdrView_;
    VkDescriptorImageInfo imgs[2] = {{samplerLinear_, srcView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                                     {samplerLinear_, bloomView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}};
    VkWriteDescriptorSet writes[2]{};
    for (int i = 0; i < 2; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = compositeSet_;
        writes[i].dstBinding = static_cast<uint32_t>(i);
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[i].pImageInfo = &imgs[i];
    }
    vkUpdateDescriptorSets(ctx_->device(), 2, writes, 0, nullptr);

    VkDescriptorImageInfo bloomSrc{samplerLinear_, srcView,
                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet bw{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    bw.dstSet = bloomSet_;
    bw.dstBinding = 0;
    bw.descriptorCount = 1;
    bw.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bw.pImageInfo = &bloomSrc;
    vkUpdateDescriptorSets(ctx_->device(), 1, &bw, 0, nullptr);
    usingResolved_ = useResolved;
}

void HairGpuRenderer::recordTaaPass(VkCommandBuffer cmd, bool reset) {
    if (taaPipeline_ == VK_NULL_HANDLE || taaSet_ == VK_NULL_HANDLE || !taaUniformMapped_)
        return;

    TaaUniforms u{};
    std::memcpy(u.prevViewProj, prevViewProj_, sizeof(u.prevViewProj));
    std::memcpy(u.invViewProj, currentInvViewProj_, sizeof(u.invViewProj));
    u.jitter[0] = jitter_[0];
    u.jitter[1] = jitter_[1];
    u.jitter[2] = prevJitter_[0];
    u.jitter[3] = prevJitter_[1];
    u.params[0] = 0.12f;                                   // blend weight (fixed jitter)
    u.params[1] = (reset || !taaHistoryValid_) ? 1.0f : 0.0f;
    std::memcpy(taaUniformMapped_, &u, sizeof(u));

    ctx_->imageLayoutTransition(cmd, resolvedImage_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                VK_IMAGE_ASPECT_COLOR_BIT);
    VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    color.imageView = resolvedView_;
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.clearValue.color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, {width_, height_}};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &color;
    vkCmdBeginRendering(cmd, &ri);
    VkViewport vp{0.0f, 0.0f, static_cast<float>(width_), static_cast<float>(height_), 0.0f, 1.0f};
    VkRect2D sc{{0, 0}, {width_, height_}};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, taaPipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, taaPipelineLayout_, 0, 1,
                            &taaSet_, 0, nullptr);
    float params[4] = {0.12f, reset ? 1.0f : 0.0f, 0.0f, 0.0f};
    vkCmdPushConstants(cmd, taaPipelineLayout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(params),
                       params);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRendering(cmd);

    // The resolved frame becomes the next frame's history, and the composite/bloom passes
    // read it instead of the raw HDR target.
    ctx_->imageLayoutTransition(cmd, resolvedImage_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
    ctx_->imageLayoutTransition(cmd, historyImage_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
    VkImageCopy copy{};
    copy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.dstSubresource = copy.srcSubresource;
    copy.extent = {width_, height_, 1};
    vkCmdCopyImage(cmd, resolvedImage_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, historyImage_,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    ctx_->imageLayoutTransition(cmd, historyImage_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                VK_IMAGE_ASPECT_COLOR_BIT);
    ctx_->imageLayoutTransition(cmd, resolvedImage_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                VK_IMAGE_ASPECT_COLOR_BIT);

    taaHistoryValid_ = true;
}

}  // namespace silk
