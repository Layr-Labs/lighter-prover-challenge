// Build-time fat metallib: AIR + this device's applegpu slice.
// No Xcode. In-process MSL compile + MTLBinaryArchive serialize.
// Ranked compile runs on the M4 Pro, so the slice matches the scored GPU.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>

static NSString *const kRequiredKernels[] = {
    @"poseidon2_hash_leaves",
    @"poseidon2_hash_leaves_colmajor",
    @"poseidon2_hash_parents",
    @"poseidon2_absorb_pass",
    @"ntt_prepare",
    @"ntt_stage",
    @"ifft_finalize",
    @"poseidon2_gate_quotient",
    @"range_check_gate_quotient",
    @"permutation_quotient",
};

int main(int argc, const char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "usage: gen_metallib SRC.metal OUT.metallib\n");
        return 2;
    }
    @autoreleasepool {
        NSError *error = nil;
        NSString *source = [NSString stringWithContentsOfFile:@(argv[1])
                                                     encoding:NSUTF8StringEncoding
                                                        error:&error];
        if (source == nil) {
            fprintf(stderr, "read source: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            fprintf(stderr, "no Metal device\n");
            return 1;
        }
        MTLCompileOptions *options = [MTLCompileOptions new];
        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
        if (library == nil) {
            fprintf(stderr, "compile: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLBinaryArchive> archive =
            [device newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new] error:&error];
        if (archive == nil) {
            fprintf(stderr, "archive: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        for (size_t i = 0; i < sizeof(kRequiredKernels) / sizeof(kRequiredKernels[0]); ++i) {
            NSString *name = kRequiredKernels[i];
            id<MTLFunction> function = [library newFunctionWithName:name];
            if (function == nil) {
                fprintf(stderr, "missing kernel %s\n", name.UTF8String);
                return 1;
            }
            MTLComputePipelineDescriptor *descriptor = [MTLComputePipelineDescriptor new];
            descriptor.computeFunction = function;
            if (![archive addComputePipelineFunctionsWithDescriptor:descriptor error:&error]) {
                fprintf(
                    stderr,
                    "add %s: %s\n",
                    name.UTF8String,
                    error.localizedDescription.UTF8String);
                return 1;
            }
        }
        NSURL *url = [NSURL fileURLWithPath:@(argv[2])];
        if (![archive serializeToURL:url error:&error]) {
            fprintf(stderr, "serialize: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }
    }
    return 0;
}
