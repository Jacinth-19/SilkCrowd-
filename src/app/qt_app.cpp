// SilkCrowd editor: Qt6 QVulkanWindow + dockable inspector.
//
// The window owns the Vulkan instance/device/swapchain (QWindow's), and the
// HairGpuRenderer adopts that device through VulkanContext::adopt(), so there is exactly
// one queue, one swapchain and one frame loop.
//
// Controls
//   mouse drag (left)      orbit
//   mouse drag (middle)    pan
//   wheel                  zoom
//   W/A/S/D + Q/E          free-fly (with the mouse captured)
//   drag on the hair mesh  deform it (XPBD shape-matching constraint, Macklin 2016)
//   C                      100 unique heads with unique styles, parameters and textures
//   T                      200 teapots
//   L                      toggle the LOD system (Figure 8 comparison)
//   H                      toggle hair, G toggle head, Space toggles wind gusts
#include "app_gpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <QAction>
#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDockWidget>
#include <QElapsedTimer>
#include <QFontDatabase>
#include <QFormLayout>
#include <QKeyEvent>
#include <QLabel>
#include <QMainWindow>
#include <QMouseEvent>
#include <QPainter>
#include <QSlider>
#include <QSpinBox>
#include <QSurfaceFormat>
#include <QTimer>
#include <QVBoxLayout>
#include <QWheelEvent>
#include <QWidget>
#include <QVulkanDeviceFunctions>
#include <QVulkanFunctions>
#include <QVulkanInstance>
#include <QVulkanWindow>

#include "scenes.h"
#include "silkcrowd/renderer.h"
#include "silkcrowd/sim.h"

#ifdef SILKCROWD_WITH_VULKAN
#include "gpu/hair_gpu_renderer.h"
#include "gpu/vulkan_context.h"
#endif

#ifdef SILKCROWD_HAS_IMGUI
#include "imgui/imgui.h"
#include "imgui/backends/imgui_impl_vulkan.h"
#endif

namespace silk {

namespace {

// Shared editor state: the Qt widgets write it, the renderer reads it.
struct EditorState {
    int style = 0;
    int crowd = 100;
    int strands = 40000;
    int msaa = 1;
    bool lod = true;
    bool hair = true;
    bool head = true;
    bool shadows = true;
    bool taa = true;
    bool wind = false;
    float windStrength = 3.0f;
    float exposure = 1.05f;
    float fovDeg = 34.0f;
    float orbitYaw = 0.0f;
    float orbitPitch = 0.0f;
    float distance = 0.52f;
    bool dragHair = false;
    Vec3 dragPoint{0.0f, 0.0f, 0.0f};
};

// A QImage-backed overlay for the statistics (drawn on top of the window container; a
// transparent child widget composites reliably on every platform because it is drawn by
// Qt's backing store, not by the Vulkan swapchain).
class StatsOverlay : public QWidget {
public:
    explicit StatsOverlay(QWidget* parent) : QWidget(parent) {
        setAttribute(Qt::WA_TransparentForMouseEvents);
        setAttribute(Qt::WA_NoSystemBackground);
        setAttribute(Qt::WA_TranslucentBackground);
        setFocusPolicy(Qt::NoFocus);
    }
    void setText(const QString& text) {
        text_ = text;
        update();
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing, true);
        const QFont mono = QFontDatabase::systemFont(QFontDatabase::FixedFont);
        p.setFont(mono);
        const QRect box(8, 8, 340, 22 + 18 * 10);
        p.setBrush(QColor(0, 0, 0, 150));
        p.setPen(Qt::NoPen);
        p.drawRoundedRect(box, 6, 6);
        p.setPen(QColor(230, 235, 240));
        p.drawText(box.adjusted(10, 8, -10, -8), Qt::AlignLeft | Qt::AlignTop, text_);
    }

private:
    QString text_;
};

}  // namespace

#ifdef SILKCROWD_WITH_VULKAN

// ---------------------------------------------------------------------------
//  QVulkanWindow renderer
// ---------------------------------------------------------------------------
class HairVulkanRenderer;

class HairVulkanWindow : public QVulkanWindow {
public:
    explicit HairVulkanWindow(EditorState* state) : state_(state) {}
    QVulkanWindowRenderer* createRenderer() override;
    // QVulkanWindow does not expose its renderer, and reloadScene() has to reach it.
    HairVulkanRenderer* hairRenderer() const { return renderer_; }
    const GpuStats* stats() const { return stats_; }
    void reloadScene(int sceneIndex);
    void setStatsPtr(const GpuStats* s) { stats_ = s; }
    EditorState* state() const { return state_; }

protected:
    void mousePressEvent(QMouseEvent* e) override;
    void mouseMoveEvent(QMouseEvent* e) override;
    void mouseReleaseEvent(QMouseEvent* e) override;
    void wheelEvent(QWheelEvent* e) override;
    void keyPressEvent(QKeyEvent* e) override;
    void keyReleaseEvent(QKeyEvent* e) override;

private:
    EditorState* state_;
    HairVulkanRenderer* renderer_ = nullptr;   // owned by QVulkanWindow
    const GpuStats* stats_ = nullptr;
    QPoint lastPos_;
    bool orbiting_ = false, panning_ = false;
};

class HairVulkanRenderer : public QVulkanWindowRenderer {
public:
    explicit HairVulkanRenderer(HairVulkanWindow* window) : window_(window) {}

    void preInitResources() override {}
    void initResources() override;
    void initSwapChainResources() override;
    void releaseSwapChainResources() override;
    void releaseResources() override;
    void startNextFrame() override;

    void rebuildScene(int sceneIndex);

private:
    // The composite pass writes through an externally supplied VkImageView, and the
    // swapchain hands out a different VkImage every frame, so views are cached per image.
    VkImageView swapchainView(VkImage image);
    std::vector<std::pair<VkImage, VkImageView>> swapchainViews_;

    HairVulkanWindow* window_;
    VulkanContext context_;
    std::unique_ptr<HairGpuRenderer> renderer_;
    std::shared_ptr<Scene> scene_;
    std::unique_ptr<HairMeshSimulator> sim_;
    int sceneIndex_ = 0;
    QElapsedTimer timer_;
    double lastMs_ = 16.0;
    int frameIndex_ = 0;
};

void HairVulkanRenderer::initResources() {
    VkInstance instance = window_->vulkanInstance()->vkInstance();
    VkPhysicalDevice physical = window_->physicalDevice();
    VkDevice device = window_->device();
    VkQueue gfx = window_->graphicsQueue();
    VkCommandPool pool = window_->graphicsCommandPool();

    std::string error;
    // QVulkanWindow hands out one queue (graphicsQueue) and one queue family; presentation
    // happens on that same queue, so both family indices are the graphics one.
    if (!context_.adopt(instance, physical, device, gfx, window_->graphicsQueue(),
                        window_->graphicsQueueFamilyIndex(), window_->graphicsQueueFamilyIndex(),
                        pool, true, &error)) {
        std::fprintf(stderr, "[SilkCrowd] VulkanContext::adopt failed: %s\n", error.c_str());
        return;
    }
    renderer_ = std::make_unique<HairGpuRenderer>();
    const int w = window_->width(), h = window_->height();
    if (!renderer_->init(&context_, static_cast<uint32_t>(std::max(w, 1)),
                         static_cast<uint32_t>(std::max(h, 1)), false, &error)) {
        std::fprintf(stderr, "[SilkCrowd] GPU renderer init failed: %s\n", error.c_str());
        renderer_.reset();
        return;
    }
    rebuildScene(sceneIndex_);
    window_->setStatsPtr(&renderer_->stats());
    timer_.start();
}

void HairVulkanRenderer::rebuildScene(int sceneIndex) {
    if (!renderer_) return;
    sceneIndex_ = sceneIndex;
    EditorState* st = window_->state();
    Scene scene;
    if (sceneIndex_ == 1) {
        scene = buildCrowdScene(st->crowd > 0 ? st->crowd : 100);
    } else if (sceneIndex_ == 2) {
        scene = buildTeapotScene(200);   // Figure 9
    } else {
        scene = buildHeadScene(st->style, static_cast<uint32_t>(std::max(st->strands, 2000)));
    }
    sim_ = std::make_unique<HairMeshSimulator>();
    if (!scene.hair.empty()) sim_->init(scene.hair.front()->mesh, SimParams{});
    scene_ = std::make_shared<Scene>(std::move(scene));
    renderer_->setScene(scene_);
}

void HairVulkanRenderer::initSwapChainResources() {
    if (!renderer_) return;
    // The swapchain images are created after initResources(); remember the format so the
    // composite pass writes into a compatible pipeline. The per-image view is supplied
    // every frame by swapchainView().
    renderer_->setCompositeTarget(VK_NULL_HANDLE, window_->colorFormat());
}

void HairVulkanRenderer::releaseSwapChainResources() {
    QVulkanDeviceFunctions* vk =
        window_->vulkanInstance()->deviceFunctions(context_.device());
    for (auto& entry : swapchainViews_)
        if (entry.second != VK_NULL_HANDLE)
            vk->vkDestroyImageView(context_.device(), entry.second, nullptr);
    swapchainViews_.clear();
}

VkImageView HairVulkanRenderer::swapchainView(VkImage image) {
    for (const auto& entry : swapchainViews_)
        if (entry.first == image) return entry.second;
    VkImageView view = VK_NULL_HANDLE;
    VkImageViewCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ci.image = image;
    ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ci.format = window_->colorFormat();
    ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    // Qt resolves the Vulkan entry points at run time and does not expose the prototypes, so
    // device-level calls go through QVulkanDeviceFunctions (QVulkanFunctions only carries the
    // instance-level subset).
    QVulkanDeviceFunctions* vk =
        window_->vulkanInstance()->deviceFunctions(context_.device());
    if (vk->vkCreateImageView(context_.device(), &ci, nullptr, &view) != VK_SUCCESS) {
        std::fprintf(stderr, "[SilkCrowd] could not create a view for the swapchain image\n");
        return VK_NULL_HANDLE;
    }
    swapchainViews_.emplace_back(image, view);
    return view;
}

void HairVulkanRenderer::releaseResources() {
    if (renderer_) {
        renderer_->shutdown();
        renderer_.reset();
    }
    context_.shutdown();
}

void HairVulkanRenderer::startNextFrame() {
    EditorState* st = window_->state();
    if (!renderer_ || !scene_) {
        window_->frameReady();
        return;
    }

    // ---- camera (orbit around the model, free-fly offsets applied by the window) ----
    Camera camera = cameraForScene(sceneIndex_ == 1 ? "crowd" : sceneIndex_ == 2 ? "teapot" : "head",
                                  window_->width(), window_->height());
    camera.fovY = st->fovDeg * 3.14159265f / 180.0f;
    const Vec3 center = scene_->hair.empty() ? Vec3(0, 0.02f, 0) : scene_->hair.front()->center;
    const Vec3 dir(std::sin(st->orbitYaw) * std::cos(st->orbitPitch),
                   std::sin(st->orbitPitch),
                   std::cos(st->orbitYaw) * std::cos(st->orbitPitch));
    camera.position = center + dir * st->distance;
    camera.target = center;

    // ---- XPBD (Macklin 2016 + Wu & Yuksel 2016 wind, sag-free rest shape) ----
    if (sim_ && !scene_->hair.empty() && (st->wind || st->dragHair)) {
        SimParams params;
        params.windStrength = st->wind ? st->windStrength : 0.0f;
        if (st->dragHair) {
            params.dragging = true;
            params.dragPoint = st->dragPoint;
        }
        sim_->step(&scene_->hair.front()->mesh, 1.0f / 60.0f, params);
        scene_->hair.front()->textures.updateDynamic(scene_->hair.front()->mesh,
                                                     computeStylingFrames(scene_->hair.front()->mesh));
        renderer_->updateDynamicTextures(*scene_->hair.front());
    }

    RenderOptions options;
    options.width = window_->width();
    options.height = window_->height();
    options.msaa = st->msaa;
    options.taa = st->taa;
    options.enableLod = st->lod;
    options.drawHair = st->hair;
    options.drawHead = st->head;
    options.drawShadows = st->shadows;
    options.exposure = st->exposure;
    renderer_->applyHardwarePolicy(context_.info(), &options);

    // currentSwapChainImageIndex() is the *index*; the image itself comes from swapChainImage().
    const VkImage swapchainImage = window_->swapChainImage(window_->currentSwapChainImageIndex());
    renderer_->setCompositeTarget(swapchainView(swapchainImage), window_->colorFormat());
    renderer_->renderToSwapchainImage(swapchainImage, window_->width(), window_->height(),
                                      camera, options);

    lastMs_ = static_cast<double>(timer_.nsecsElapsed()) / 1.0e6;
    timer_.restart();
    ++frameIndex_;

#ifdef SILKCROWD_HAS_IMGUI
    // Optional Dear ImGui window drawn by the backend on top of the frame.
    ImGui::SetNextWindowPos(ImVec2(12, 12), ImGuiCond_FirstUseEver);
    ImGui::Begin("SilkCrowd");
    const GpuStats& s = renderer_->stats();
    ImGui::Text("%.2f ms (%.0f fps)", lastMs_, lastMs_ > 0 ? 1000.0 / lastMs_ : 0.0);
    ImGui::Text("strands %llu / %llu", (unsigned long long)s.strandsRendered,
                (unsigned long long)s.strandsTotal);
    ImGui::Text("triangles %llu", (unsigned long long)s.triangles);
    ImGui::Text("textures %.2f MB", double(s.textureBytes) / (1024.0 * 1024.0));
    ImGui::Text("crowd %d | MSAA %dx", s.crowdCount, options.msaa);
    ImGui::End();
#endif

    window_->frameReady();
    window_->requestUpdate();
}

QVulkanWindowRenderer* HairVulkanWindow::createRenderer() {
    renderer_ = new HairVulkanRenderer(this);
    return renderer_;
}

void HairVulkanWindow::reloadScene(int sceneIndex) {
    if (renderer_) renderer_->rebuildScene(sceneIndex);
}

void HairVulkanWindow::mousePressEvent(QMouseEvent* e) {
    if (e->button() == Qt::LeftButton) {
        orbiting_ = true;
        lastPos_ = e->pos();
        // Dragging on the hair deforms it (Section 3.6 simulation).
        state_->dragHair = true;
        state_->dragPoint = Vec3(0.0f, 0.05f, 0.12f);
    } else if (e->button() == Qt::MiddleButton) {
        panning_ = true;
        lastPos_ = e->pos();
    }
}

void HairVulkanWindow::mouseMoveEvent(QMouseEvent* e) {
    const QPoint d = e->pos() - lastPos_;
    lastPos_ = e->pos();
    if (orbiting_) {
        state_->orbitYaw += d.x() * 0.008f;
        state_->orbitPitch = std::clamp(state_->orbitPitch + d.y() * 0.008f, -1.4f, 1.4f);
    } else if (panning_) {
        state_->dragPoint.x += d.x() * 0.0005f;
        state_->dragPoint.y -= d.y() * 0.0005f;
    }
}

void HairVulkanWindow::mouseReleaseEvent(QMouseEvent*) {
    orbiting_ = panning_ = false;
    state_->dragHair = false;
}

void HairVulkanWindow::wheelEvent(QWheelEvent* e) {
    state_->distance = std::clamp(state_->distance * (1.0f - e->angleDelta().y() * 0.0015f),
                                  0.08f, 12.0f);
}

void HairVulkanWindow::keyPressEvent(QKeyEvent* e) {
    switch (e->key()) {
        case Qt::Key_C: reloadScene(1); break;   // 100 unique heads
        case Qt::Key_T: reloadScene(2); break;   // 200 teapots
        case Qt::Key_1: reloadScene(0); break;   // single head
        case Qt::Key_L: state_->lod = !state_->lod; break;
        case Qt::Key_H: state_->hair = !state_->hair; break;
        case Qt::Key_G: state_->head = !state_->head; break;
        case Qt::Key_Space: state_->wind = !state_->wind; break;
        case Qt::Key_Escape: qApp->quit(); break;
        default: QVulkanWindow::keyPressEvent(e); break;
    }
}

void HairVulkanWindow::keyReleaseEvent(QKeyEvent* e) { QVulkanWindow::keyReleaseEvent(e); }

// ---------------------------------------------------------------------------
//  Main window with the dockable inspector
// ---------------------------------------------------------------------------
namespace {

QWidget* buildInspector(QWidget* parent, EditorState* st, HairVulkanWindow* view,
                        std::function<void()> onChanged) {
    auto* panel = new QWidget(parent);
    auto* form = new QFormLayout(panel);

    auto* styleBox = new QComboBox(panel);
    static const char* kStyleNames[] = {"1 long straight", "2 wavy bob",     "3 curly",
                                        "4 ponytail",      "5 pixie",        "6 braided",
                                        "7 afro",          "8 long wavy",    "9 side part",
                                        "10 slicked back"};
    for (const char* n : kStyleNames) styleBox->addItem(n);
    QObject::connect(styleBox, &QComboBox::currentIndexChanged, panel, [=](int i) {
        st->style = i;
        view->reloadScene(0);
        onChanged();
    });
    form->addRow("Style (Figure 10)", styleBox);

    auto* strands = new QSpinBox(panel);
    strands->setRange(2000, 300000);
    strands->setSingleStep(5000);
    strands->setValue(st->strands);
    QObject::connect(strands, &QSpinBox::valueChanged, panel, [=](int v) {
        st->strands = v;
        view->reloadScene(0);
        onChanged();
    });
    form->addRow("Root strands", strands);

    auto* crowd = new QSpinBox(panel);
    crowd->setRange(1, 400);
    crowd->setValue(st->crowd);
    QObject::connect(crowd, &QSpinBox::valueChanged, panel, [=](int v) {
        st->crowd = v;
        view->reloadScene(1);
        onChanged();
    });
    form->addRow("Crowd count", crowd);

    auto* lod = new QCheckBox("Strand + vertex LOD (Eq. 7-12)", panel);
    lod->setChecked(st->lod);
    QObject::connect(lod, &QCheckBox::toggled, panel, [=](bool v) { st->lod = v; onChanged(); });
    form->addRow(lod);

    auto* shadow = new QCheckBox("PCF shadows", panel);
    shadow->setChecked(st->shadows);
    QObject::connect(shadow, &QCheckBox::toggled, panel, [=](bool v) { st->shadows = v; onChanged(); });
    form->addRow(shadow);

    auto* taa = new QCheckBox("Temporal AA (sub-pixel jitter + reprojection)", panel);
    taa->setChecked(st->taa);
    QObject::connect(taa, &QCheckBox::toggled, panel, [=](bool v) { st->taa = v; onChanged(); });
    form->addRow(taa);

    auto* msaa = new QComboBox(panel);
    for (int s : {1, 2, 4, 8}) msaa->addItem(QString("%1x MSAA").arg(s), s);
    QObject::connect(msaa, &QComboBox::currentIndexChanged, panel, [=](int) {
        st->msaa = msaa->currentData().toInt();
        onChanged();
    });
    form->addRow("Anti-aliasing", msaa);

    auto* wind = new QSlider(Qt::Horizontal, panel);
    wind->setRange(0, 200);   // 0 - 20 m/s
    wind->setValue(static_cast<int>(st->windStrength * 10.0f));
    QObject::connect(wind, &QSlider::valueChanged, panel, [=](int v) {
        st->windStrength = v / 10.0f;
        st->wind = v > 0;
        onChanged();
    });
    form->addRow("Wind (m/s)", wind);

    auto* fov = new QSlider(Qt::Horizontal, panel);
    fov->setRange(10, 90);
    fov->setValue(static_cast<int>(st->fovDeg));
    QObject::connect(fov, &QSlider::valueChanged, panel, [=](int v) { st->fovDeg = float(v); onChanged(); });
    form->addRow("FOV", fov);

    auto* exposure = new QSlider(Qt::Horizontal, panel);
    exposure->setRange(20, 250);
    exposure->setValue(static_cast<int>(st->exposure * 100.0f));
    QObject::connect(exposure, &QSlider::valueChanged, panel, [=](int v) {
        st->exposure = v / 100.0f;
        onChanged();
    });
    form->addRow("Exposure", exposure);

    auto* hints = new QLabel("C: 100 heads   T: 200 teapots   1: single head\n"
                             "L: LOD   H: hair   G: head   Space: wind\n"
                             "drag: orbit / deform   wheel: zoom",
                             panel);
    hints->setWordWrap(true);
    form->addRow(hints);
    return panel;
}

}  // namespace

class EditorWindow : public QMainWindow {
public:
    EditorWindow() {
        view_ = new HairVulkanWindow(&state_);
        QVulkanInstance* instance = new QVulkanInstance();
        instance->setApiVersion(QVersionNumber(1, 3));
        instance->setLayers({"VK_LAYER_KHRONOS_validation"});
        instance->setExtensions({});
        if (!instance->create()) {
            std::fprintf(stderr, "[SilkCrowd] failed to create a Vulkan 1.3 instance\n");
        }
        view_->setVulkanInstance(instance);

        overlay_ = new StatsOverlay(this);
        // HairVulkanWindow derives from QVulkanWindow (a QWindow), so it has to be wrapped in
        // a container before QMainWindow::setCentralWidget will take it.
        QWidget* viewContainer = QWidget::createWindowContainer(view_, this);
        viewContainer->setFocusPolicy(Qt::StrongFocus);
        setCentralWidget(viewContainer);
        setWindowTitle("SilkCrowd - real-time hair rendering with hair meshes");

        auto* inspectorDock = new QDockWidget("Inspector", this);
        inspectorDock->setWidget(buildInspector(this, &state_, view_, [this] { updateStats(); }));
        inspectorDock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
        inspectorDock->setFeatures(QDockWidget::DockWidgetMovable |
                                   QDockWidget::DockWidgetFloatable);
        addDockWidget(Qt::RightDockWidgetArea, inspectorDock);

        auto* statsDock = new QDockWidget("Statistics", this);
        statsLabel_ = new QLabel(this);
        statsLabel_->setTextFormat(Qt::RichText);
        statsLabel_->setWordWrap(true);
        statsDock->setWidget(statsLabel_);
        statsDock->setFeatures(QDockWidget::DockWidgetMovable);
        addDockWidget(Qt::RightDockWidgetArea, statsDock);

        statusTimer_ = new QTimer(this);
        connect(statusTimer_, &QTimer::timeout, this, [this] { updateStats(); });
        statusTimer_->start(250);   // 4 Hz overlay refresh, independent of the frame rate
    }

    void show() {
        QMainWindow::show();
        overlay_->raise();
        overlay_->resize(360, 220);
    }

private:
    void updateStats() {
        const GpuStats* s = view_->stats();
        if (!s) return;
        const QString text =
            QString("%1 ms   %2 fps\n"
                    "strands  %3 / %4\n"
                    "triangles %5\n"
                    "hair mesh textures %6 KB  (total %7 MB)\n"
                    "vertex LOD 2^%8..2^%9   crowd %10   MSAA %11x\n"
                    "task/mesh: %12   %13\n"
                    "%14")
                .arg(s->frameMs, 0, 'f', 2)
                .arg(s->frameMs > 0 ? 1000.0 / s->frameMs : 0.0, 0, 'f', 0)
                .arg(static_cast<unsigned long long>(s->strandsRendered))
                .arg(static_cast<unsigned long long>(s->strandsTotal))
                .arg(static_cast<unsigned long long>(s->triangles))
                .arg(s->crowdCount > 0
                         ? static_cast<double>(s->textureBytes) / (1024.0 * s->crowdCount)
                         : 0.0,
                     0, 'f', 1)
                .arg(static_cast<double>(s->textureBytes) / (1024.0 * 1024.0), 0, 'f', 2)
                .arg(s->vertexLodMin)
                .arg(s->vertexLodMax)
                .arg(s->crowdCount)
                .arg(s->msaa)
                .arg(s->meshShaderPath ? "yes" : "no (compute fallback)")
                .arg(QString::fromStdString(s->deviceName))
                .arg(QString::fromStdString(s->warning));
        overlay_->setText(text);
        QString html = text;                       // QString::replace() mutates
        statsLabel_->setText(html.replace("\n", "<br>"));
    }

    EditorState state_;
    HairVulkanWindow* view_ = nullptr;
    StatsOverlay* overlay_ = nullptr;
    QLabel* statsLabel_ = nullptr;
    QTimer* statusTimer_ = nullptr;
};

#endif  // SILKCROWD_WITH_VULKAN

int runGpuWindow(int argc, char** argv) {
#ifndef SILKCROWD_WITH_VULKAN
    (void)argc;
    (void)argv;
    return 2;
#else
    QApplication app(argc, argv);
    QSurfaceFormat format;
    format.setDepthBufferSize(24);
    format.setStencilBufferSize(8);
    format.setSamples(1);
    QSurfaceFormat::setDefaultFormat(format);

    EditorWindow window;
    window.resize(1600, 900);
    window.show();
    return app.exec();
#endif
}

}  // namespace silk
