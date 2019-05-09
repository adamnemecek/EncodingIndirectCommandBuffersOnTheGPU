/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Implementation of our cross-platform view controller
*/

#import "AAPLViewController.h"
#import "AAPLRenderer.h"

@implementation AAPLViewController
{
    MTKView *_view;

    AAPLRenderer *_renderer;
}

- (void)viewDidLoad
{
    [super viewDidLoad];

    // Set the view to use the default device
    _view = (MTKView *)self.view;
    _view.device = MTLCreateSystemDefaultDevice();

    if(!_view.device)
    {
        NSLog(@"Metal is not supported on this device");
        return;
    }
    
    BOOL sampleSupported = NO;
#if TARGET_IOS
    sampleSupported = [_view.device supportsFeatureSet:MTLFeatureSet_iOS_GPUFamily4_v2];
#else
    sampleSupported = [_view.device supportsFeatureSet:MTLFeatureSet_macOS_GPUFamily2_v1];
#endif
    if (!sampleSupported)
    {
        NSLog(@"Sample requires iOS_GPUFamily4_v2 or macOS_GPUFamily2_v1 device, or later");
        assert(!"Sample requires iOS_GPUFamily4_v2 or macOS_GPUFamily2_v1 device, or later\n");
        return;
    }

    _renderer = [[AAPLRenderer alloc] initWithMetalKitView:_view];

    if(!_renderer)
    {
        NSLog(@"Renderer failed initialization");
        return;
    }

    // Initialize our renderer with the view size
    [_renderer mtkView:_view drawableSizeWillChange:_view.drawableSize];

    _view.delegate = _renderer;
}

@end
