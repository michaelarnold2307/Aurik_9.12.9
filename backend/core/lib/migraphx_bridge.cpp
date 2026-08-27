// migraphx_bridge.cpp — C-Bridge zwischen Aurik (ctypes) und AMD MIGraphX 2.15
// (ROCm 7.2.4). Build: scripts/build_migraphx_bridge.sh
//
// ABI (ctypes-Bindings siehe backend/core/migraphx_adapter.py):
//   void* mgx_load_onnx(path, default_dim_value, shape_hints)
//   void  mgx_destroy(handle)
//   int   mgx_get_input_count(handle)
//   const char*  mgx_get_input_name(handle, index)
//   int   mgx_get_input_ndim(handle, index)
//   const int64_t* mgx_get_input_shape(handle, index)
//   int   mgx_run(handle, input_count, names, datas, shapes, ndims,
//                 float** out, int64_t out_shape[8], int* out_ndim)
//
// Rev. 2026-08-16: Neu geschrieben gegen MIGraphX 2.15 / ROCm 7.2.4 (C++-API
// mit Handle-Wrappern). Die Vorgänger-Bridge (v3, ROCm 6.2, Quelle lag in
// /tmp und ist verloren) nutzte die gleiche ABI; das Verhalten ist identisch:
//   - "main:"-Parameter werden automatisch mit Nullen gefüllt
//   - Inputs werden als Host-Puffer übergeben (eval überträgt H2D)
//   - Output wird per hipMemcpy DeviceToHost geholt
//
// LicenseRef: MIT (MIGraphX-Header-Lizenz gilt für die API-Nutzung).

#include <migraphx/migraphx.hpp>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Session {
    migraphx::program prog;
    std::vector<std::string> param_names;                       // sortiert → deterministisch
    std::vector<migraphx::shape> param_shapes;
    std::vector<std::vector<int64_t>> param_lens;               // int64 für ctypes
    std::vector<float> out_buf;                                 // wiederverwendeter CPU-Puffer
    std::mutex mtx;
};

// Shape-Hints parsen: "name:d0,d1;name2:d0,d1" → options.set_input_parameter_shape
void apply_shape_hints(migraphx::onnx_options& options, const char* shape_hints)
{
    if(shape_hints == nullptr || shape_hints[0] == '\0')
        return;
    std::string hints(shape_hints);
    size_t pos = 0;
    while(pos < hints.size())
    {
        size_t sep     = hints.find(';', pos);
        std::string item =
            hints.substr(pos, sep == std::string::npos ? std::string::npos : sep - pos);
        size_t colon = item.find(':');
        if(colon != std::string::npos)
        {
            std::string name      = item.substr(0, colon);
            std::string dims_part = item.substr(colon + 1);
            std::vector<std::size_t> dims;
            std::stringstream ss(dims_part);
            std::string tok;
            while(std::getline(ss, tok, ','))
            {
                if(not tok.empty())
                    dims.push_back(static_cast<std::size_t>(std::stoul(tok)));
            }
            if(not name.empty() and not dims.empty())
                options.set_input_parameter_shape(name, dims);
        }
        if(sep == std::string::npos)
            break;
        pos = sep + 1;
    }
}

int copy_shape_to(const migraphx::shape& s, std::vector<int64_t>& out)
{
    const size_t* lens  = nullptr;
    size_t nd           = 0;
    migraphx_shape_lengths(&lens, &nd, s.get_handle_ptr());
    out.assign(lens, lens + nd);
    return static_cast<int>(nd);
}

} // namespace

extern "C" {

void* mgx_load_onnx(const char* path, size_t default_dim_value, const char* shape_hints)
{
    try
    {
        migraphx::onnx_options options;
        if(default_dim_value > 0)
            options.set_default_dim_value(static_cast<unsigned int>(default_dim_value));
        apply_shape_hints(options, shape_hints);

        auto* s  = new Session();
        s->prog  = migraphx::parse_onnx(path, options);
        // offload_copy=true: Host↔Device-Kopien laufen im Programm selbst —
        // einziger C-API-kompatibler Weg, Ergebnisse auf dem Host zu bekommen
        // (t.copy_to/t.copy_from sind in der C-API nicht exponiert, Rev. 2026-08-16).
        migraphx::compile_options copts;
        copts.set_offload_copy(true);
        s->prog.compile(migraphx::target("gpu"), copts);

        auto pshapes = s->prog.get_parameter_shapes();
        auto names   = pshapes.names();
        std::vector<std::string> sorted(names.begin(), names.end());
        std::sort(sorted.begin(), sorted.end());
        s->param_names.reserve(sorted.size());
        s->param_shapes.reserve(sorted.size());
        s->param_lens.reserve(sorted.size());
        for(const auto& n : sorted)
        {
            migraphx::shape shp = pshapes[n.c_str()];
            s->param_names.push_back(n);
            s->param_shapes.push_back(shp);
            s->param_lens.emplace_back();
            copy_shape_to(shp, s->param_lens.back());
        }
        return s;
    }
    catch(const std::exception&)
    {
        return nullptr;
    }
}

void mgx_destroy(void* handle)
{
    if(handle == nullptr)
        return;
    auto* s = static_cast<Session*>(handle);
    {
        std::lock_guard<std::mutex> lock(s->mtx);
    }
    delete s;
}

int mgx_get_input_count(void* handle)
{
    if(handle == nullptr)
        return 0;
    auto* s = static_cast<Session*>(handle);
    return static_cast<int>(s->param_names.size());
}

const char* mgx_get_input_name(void* handle, int index)
{
    if(handle == nullptr || index < 0)
        return nullptr;
    auto* s = static_cast<Session*>(handle);
    if(static_cast<size_t>(index) >= s->param_names.size())
        return nullptr;
    return s->param_names[static_cast<size_t>(index)].c_str();
}

int mgx_get_input_ndim(void* handle, int index)
{
    if(handle == nullptr || index < 0)
        return 0;
    auto* s = static_cast<Session*>(handle);
    if(static_cast<size_t>(index) >= s->param_lens.size())
        return 0;
    return static_cast<int>(s->param_lens[static_cast<size_t>(index)].size());
}

const int64_t* mgx_get_input_shape(void* handle, int index)
{
    if(handle == nullptr || index < 0)
        return nullptr;
    auto* s = static_cast<Session*>(handle);
    if(static_cast<size_t>(index) >= s->param_lens.size())
        return nullptr;
    return s->param_lens[static_cast<size_t>(index)].data();
}

int mgx_run(void* handle,
            int input_count,
            const char* const* input_names,
            const float* const* input_data,
            const int64_t* const* /*input_shapes*/,
            const int* /*input_ndims*/,
            float** output_data,
            int64_t* output_shape,
            int* output_ndim)
{
    if(handle == nullptr || output_data == nullptr || output_shape == nullptr ||
       output_ndim == nullptr)
        return 1;
    auto* s = static_cast<Session*>(handle);
    std::lock_guard<std::mutex> lock(s->mtx);
    try
    {
        // MIGraphX 2.15 (ROCm 7.2.4): mit offload_copy=true verwaltet das
        // gpu-Programm die Host↔Device-Kopien intern (Parameter bleiben HOST-
        // Puffer, Ergebnisse kommen als Host-Argumente zurück). Der 2.15-Treiber
        // nutzt für diesen Pfad t.copy_to/t.copy_from — die die C-API nicht
        // exponiert; offload_copy ist der C-API-kompatible Weg (Rev. 2026-08-16).
        migraphx::program_parameters params;
        for(size_t i = 0; i < s->param_names.size(); ++i)
        {
            const std::string& name = s->param_names[i];
            const migraphx::shape& shp = s->param_shapes[i];
            size_t bytes = shp.bytes();

            const float* src = nullptr;
            for(int j = 0; j < input_count; ++j)
            {
                if(input_names[j] != nullptr && name == input_names[j])
                {
                    src = input_data[j];
                    break;
                }
            }

            migraphx::argument arg(shp);
            if(src != nullptr)
                std::memcpy(arg.data(), src, bytes);
            else
                std::memset(arg.data(), 0, bytes); // interne Puffer (z.B. "main:") genullt
            params.add(name.c_str(), arg);
        }

        auto results = s->prog.eval(params);
        if(results.size() == 0)
            return 2;

        auto out       = results[0];
        auto out_shape = out.get_shape();
        size_t nbytes  = out_shape.bytes();

        size_t nfloats = (nbytes + sizeof(float) - 1) / sizeof(float);
        if(s->out_buf.size() < nfloats)
            s->out_buf.resize(nfloats);

        // Ergebnis kann je nach MIGraphX-Pfad auf der GPU (Device) oder schon
        // auf dem Host liegen → Pointer-Typ prüfen und passend kopieren.
        // (Rev. 2026-08-16: hipMemcpy schlug auf 7.2.4 fehl, weil das Ergebnis
        // bereits ein Host-Puffer war; hipPointerGetAttributes entscheidet.)
        hipPointerAttribute_t attr;
        hipError_t perr = hipPointerGetAttributes(&attr, out.data());
        if(perr == hipSuccess && attr.type == hipMemoryTypeDevice)
        {
            hipError_t err =
                hipMemcpy(s->out_buf.data(), out.data(), nbytes, hipMemcpyDeviceToHost);
            if(err != hipSuccess)
                return 3;
        }
        else
        {
            std::memcpy(s->out_buf.data(), out.data(), nbytes);
        }

        *output_data = s->out_buf.data();
        // out_shape (max 8 dims) direkt aus dem Ergebnis befüllen
        const size_t* lens = nullptr;
        size_t nd_out      = 0;
        migraphx_shape_lengths(&lens, &nd_out, out_shape.get_handle_ptr());
        int k = static_cast<int>(std::min<size_t>(nd_out, 8));
        for(int i = 0; i < k; ++i)
            output_shape[i] = static_cast<int64_t>(lens[i]);
        *output_ndim = k;
        return 0;
    }
    catch(const std::exception& e)
    {
        fprintf(stderr, "[migraphx_bridge] mgx_run error: %s\n", e.what());
        return 1;
    }
}

} // extern "C"
