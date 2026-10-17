// AppWindow — UI thread host.
//
// Milestone 4 adds the Dear ImGui dashboard: input/output device dropdowns,
// translation ON/OFF, Wait-k slider (k in [1..10]), .onnx voice-model loading,
// and live latency / underrun telemetry fed by AudioEngine's atomic counters.
//
// Until then this translation unit is a placeholder so the build layout
// matches the documented repository structure.
namespace fluxvoice::ui {

const char* app_title() { return "FluxVoice-S2UT"; }

} // namespace fluxvoice::ui
