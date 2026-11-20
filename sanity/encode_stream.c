/* GPU regression for input readiness and output-frame reuse dependencies.
 * Build from the project root (adjust the CUDA and driver library paths):
 * cc -O2 -ffunction-sections -fdata-sections -Iinclude -I/opt/cuda/include \
 *   sanity/encode_stream.c -Wl,--gc-sections \
 *   $(pkg-config --cflags --libs libavformat libavcodec libavfilter libavutil libswscale) \
 *   -L/opt/cuda/lib64 -lcudart -o /tmp/aji_encode_stream_test
 * Run: /tmp/aji_encode_stream_test
 * Negative control: /tmp/aji_encode_stream_test --omit-input-wait
 * The negative control should fail with stale pixels. No models are required.
 */
#define main aji_encode_cli_main
#include "../src/encode.c"
#undef main

#define TEST_FRAMES 64
#define FRAME_BYTES 256

/* Delays only this producer stream; callbacks must not call CUDA APIs. */
static void CUDART_CB delay_producer(void *opaque)
{
    (void)opaque;
    av_usleep(1000);
}

static int check_stream(int custom_producer, int omit_wait, int reuse_output)
{
    enc_ctx c = {0};
    cudaStream_t producer = NULL, unrelated = NULL;
    unsigned char *input = NULL, *output = NULL, *host = NULL;
    const size_t bytes = TEST_FRAMES * FRAME_BYTES;
    int result = 1;

    /* A decoded/uploaded frame can belong to another device wrapper sharing
     * the same CUDA context. Its producer stream is the dependency to track. */
    AVCUDADeviceContext producer_cuda = {0}, own_cuda = {0};
    AVHWDeviceContext producer_device = {
        .type = AV_HWDEVICE_TYPE_CUDA, .hwctx = &producer_cuda,
    };
    AVHWDeviceContext own_device = {
        .type = AV_HWDEVICE_TYPE_CUDA, .hwctx = &own_cuda,
    };
    AVBufferRef own_ref = {.data = (uint8_t *)&own_device};
    AVHWFramesContext frames = {.device_ctx = &producer_device};
    AVBufferRef frames_ref = {.data = (uint8_t *)&frames};
    AVFrame frame = {.format = AV_PIX_FMT_CUDA, .hw_frames_ctx = &frames_ref};
    c.hw_device = &own_ref;

    CK(cudaStreamCreateWithFlags(&c.stream, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&unrelated, cudaStreamNonBlocking));
    if (custom_producer)
        CK(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking));
    producer_cuda.stream = (CUstream)producer;
    own_cuda.stream = (CUstream)unrelated;
    CK(cudaEventCreateWithFlags(&c.input_ready, cudaEventDisableTiming));
    CK(cudaMalloc((void **)&input, bytes));
    CK(cudaMalloc((void **)&output, bytes));
    CK(cudaMallocHost((void **)&host, bytes));
    CK(cudaMemsetAsync(input, 0, bytes, producer));
    CK(cudaMemsetAsync(output, 0, bytes, producer));
    CK(cudaStreamSynchronize(producer));

    /* Re-record one event for many queued frames without synchronizing the
     * host between them. Each consumer must observe its own completed write. */
    for (int i = 0; i < TEST_FRAMES; i++) {
        size_t offset = (size_t)i * FRAME_BYTES;
        if (reuse_output) {
            /* Like a reclaimed inference output passed to scale_cuda: its
             * pixels are ready, but an async reader must finish before the
             * same pool allocation can be overwritten by future inference. */
            CK(cudaMemsetAsync(input, i + 1, FRAME_BYTES, c.stream));
            CK(cudaStreamSynchronize(c.stream));
            CK(cudaLaunchHostFunc(producer, delay_producer, NULL));
            CK(cudaMemcpyAsync(output + offset, input, FRAME_BYTES,
                               cudaMemcpyDeviceToDevice, producer));
            if (!omit_wait && wait_frame_stream(&c, &frame) < 0)
                goto fail;
        } else {
            CK(cudaLaunchHostFunc(producer, delay_producer, NULL));
            CK(cudaMemsetAsync(input + offset, i + 1, FRAME_BYTES, producer));
            if (!omit_wait && wait_decoded_frame(&c, &frame) < 0)
                goto fail;
            CK(cudaMemcpyAsync(output + offset, input + offset, FRAME_BYTES,
                               cudaMemcpyDeviceToDevice, c.stream));
        }
    }
    if (reuse_output) CK(cudaStreamSynchronize(producer));
    CK(cudaMemcpyAsync(host, output, bytes, cudaMemcpyDeviceToHost, c.stream));
    CK(cudaStreamSynchronize(c.stream));

    int stale = 0;
    for (int i = 0; i < TEST_FRAMES; i++) {
        for (int j = 0; j < FRAME_BYTES; j++) {
            if (host[i * FRAME_BYTES + j] != i + 1) {
                stale++;
                break;
            }
        }
    }
    printf("%s: %s producer, %s, %d frames, %d stale frames%s\n",
           stale ? "FAIL" : "PASS", custom_producer ? "custom" : "default",
           reuse_output ? "output reuse" : "input readiness",
           TEST_FRAMES, stale, omit_wait ? " (input wait omitted)" : "");
    result = stale != 0;

fail:
    /* Also protect storage when a failed submission leaves producer work. */
    cudaStreamSynchronize(producer);
    if (c.stream) cudaStreamSynchronize(c.stream);
    if (host) cudaFreeHost(host);
    if (output) cudaFree(output);
    if (input) cudaFree(input);
    if (c.input_ready) cudaEventDestroy(c.input_ready);
    if (producer) cudaStreamDestroy(producer);
    if (unrelated) cudaStreamDestroy(unrelated);
    if (c.stream) cudaStreamDestroy(c.stream);
    return result;
}

int main(int argc, char **argv)
{
    int omit_wait = argc == 2 && !strcmp(argv[1], "--omit-input-wait");
    if (argc != 1 && !omit_wait) {
        fprintf(stderr, "usage: %s [--omit-input-wait]\n", argv[0]);
        return 2;
    }
    int result = 0;
    for (int custom = 0; custom < 2; custom++)
        for (int reuse = 0; reuse < 2; reuse++)
            result |= check_stream(custom, omit_wait, reuse);
    return result;
}
