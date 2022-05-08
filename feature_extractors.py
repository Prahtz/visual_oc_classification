from esvit_swin import EsVitPreTrained

def EsVitBase(num_blocks=4):
    checkpoint_path = 'checkpoints/swin_base_w14/checkpoint_best.pth'
    arch = 'swin_base'
    cfg = 'esvit/experiments/imagenet/swin/swin_base_patch4_window14_224.yaml'
    return EsVitPreTrained(cfg, arch, checkpoint_path, num_blocks=num_blocks, trainable=False)

def EsVitTiny(num_blocks=4):
    checkpoint_path = 'checkpoints/swin_tiny_w7/checkpoint_best.pth'
    arch = 'swin_tiny'
    cfg = 'esvit/experiments/imagenet/swin/swin_tiny_patch4_window7_224.yaml'
    return EsVitPreTrained(cfg, arch, checkpoint_path, num_blocks=num_blocks, trainable=False)