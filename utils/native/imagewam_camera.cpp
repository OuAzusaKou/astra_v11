// The legacy FFmpeg single-frame filter/encode/decode pipeline, in-process.
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

static void checked(int result) {
    if (result < 0) {
        char error[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(result, error, sizeof(error));
        throw std::runtime_error(error);
    }
}
struct GraphFree { void operator()(AVFilterGraph* p) const { avfilter_graph_free(&p); } };
struct FrameFree { void operator()(AVFrame* p) const { av_frame_free(&p); } };
struct CodecFree { void operator()(AVCodecContext* p) const { avcodec_free_context(&p); } };
struct PacketFree { void operator()(AVPacket* p) const { av_packet_free(&p); } };
using Frame = std::unique_ptr<AVFrame, FrameFree>;
using Codec = std::unique_ptr<AVCodecContext, CodecFree>;

static int process_camera(const uint8_t* pixels, int width, int height,
                          int crop, uint8_t* output, char* error, int error_size, bool bgr) {
    try {
        std::unique_ptr<AVFilterGraph, GraphFree> graph(avfilter_graph_alloc());
        graph->nb_threads = 1;
        AVFilterContext *source, *sink;
        std::string args = "video_size=" + std::to_string(width) + "x" + std::to_string(height)
            + ":pix_fmt=" + std::to_string(AV_PIX_FMT_RGB24) + ":time_base=1/25:pixel_aspect=1/1";
        checked(avfilter_graph_create_filter(&source, avfilter_get_by_name("buffer"), "in", args.c_str(), nullptr, graph.get()));
        checked(avfilter_graph_create_filter(&sink, avfilter_get_by_name("buffersink"), "out", nullptr, nullptr, graph.get()));
        std::string filters;
        if (crop < 100) {
            const std::string ratio = std::to_string(crop / 100.0);
            filters = "crop=trunc(iw*" + ratio + "/2)*2:trunc(ih*" + ratio + "/2)*2:(iw-ow)/2:(ih-oh)/2,";
        }
        filters += "scale=640:480:force_original_aspect_ratio=increase,crop=640:480,format=pix_fmts=yuv420p";
        AVFilterInOut *inputs = avfilter_inout_alloc(), *outputs = avfilter_inout_alloc();
        outputs->name = av_strdup("in"); outputs->filter_ctx = source; outputs->pad_idx = 0;
        inputs->name = av_strdup("out"); inputs->filter_ctx = sink; inputs->pad_idx = 0;
        int parsed = avfilter_graph_parse_ptr(graph.get(), filters.c_str(), &inputs, &outputs, nullptr);
        avfilter_inout_free(&inputs); avfilter_inout_free(&outputs);
        checked(parsed);
        checked(avfilter_graph_config(graph.get(), nullptr));
        Frame input(av_frame_alloc()), scaled(av_frame_alloc()), decoded(av_frame_alloc());
        input->format = AV_PIX_FMT_RGB24; input->width = width; input->height = height; input->pts = 0;
        checked(av_frame_get_buffer(input.get(), 32));
        for (int y = 0; y < height; ++y) {
            auto* dst = input->data[0] + y * input->linesize[0];
            const auto* src = pixels + y * width * 3;
            if (bgr) {
                for (int x = 0; x < width; ++x) {
                    dst[3*x] = src[3*x+2];
                    dst[3*x+1] = src[3*x+1];
                    dst[3*x+2] = src[3*x];
                }
            } else {
                std::memcpy(dst, src, width * 3);
            }
        }
        checked(av_buffersrc_add_frame_flags(source, input.get(), AV_BUFFERSRC_FLAG_KEEP_REF));
        checked(av_buffersink_get_frame(sink, scaled.get()));
        Codec encoder(avcodec_alloc_context3(avcodec_find_encoder_by_name("libx264")));
        encoder->width = 640; encoder->height = 480; encoder->pix_fmt = AV_PIX_FMT_YUV420P;
        encoder->time_base = AVRational{1, 25}; encoder->framerate = AVRational{25, 1};
        encoder->sample_aspect_ratio = scaled->sample_aspect_ratio;
        encoder->gop_size = 2; encoder->thread_count = 1;
        checked(av_opt_set(encoder->priv_data, "crf", "20", 0));
        checked(avcodec_open2(encoder.get(), nullptr, nullptr));
        checked(avcodec_send_frame(encoder.get(), scaled.get()));
        checked(avcodec_send_frame(encoder.get(), nullptr));
        std::unique_ptr<AVPacket, PacketFree> packet(av_packet_alloc());
        checked(avcodec_receive_packet(encoder.get(), packet.get()));
        Codec decoder(avcodec_alloc_context3(avcodec_find_decoder(AV_CODEC_ID_H264)));
        decoder->thread_count = 1;
        checked(avcodec_open2(decoder.get(), nullptr, nullptr));
        checked(avcodec_send_packet(decoder.get(), packet.get()));
        checked(avcodec_send_packet(decoder.get(), nullptr));
        checked(avcodec_receive_frame(decoder.get(), decoded.get()));
        SwsContext* sws = sws_getContext(640, 480, static_cast<AVPixelFormat>(decoded->format),
                                       640, 480, AV_PIX_FMT_RGB24, SWS_BICUBIC, nullptr, nullptr, nullptr);
        uint8_t* dst[] = {output, nullptr, nullptr, nullptr};
        int lines[] = {640 * 3, 0, 0, 0};
        int result = sws_scale(sws, decoded->data, decoded->linesize, 0, 480, dst, lines);
        sws_freeContext(sws);
        checked(result);
        return 0;
    } catch (const std::exception& exc) {
        std::snprintf(error, error_size, "%s", exc.what());
        return -1;
    }
}
extern "C" int imagewam_camera(const uint8_t* rgb, int width, int height,
                               int crop, uint8_t* output, char* error, int error_size) {
    return process_camera(rgb, width, height, crop, output, error, error_size, false);
}
extern "C" int imagewam_camera_bgr(const uint8_t* bgr, int width, int height,
                                   int crop, uint8_t* output, char* error, int error_size) {
    return process_camera(bgr, width, height, crop, output, error, error_size, true);
}
extern "C" void imagewam_camera_init() { av_log_set_level(AV_LOG_ERROR); }
