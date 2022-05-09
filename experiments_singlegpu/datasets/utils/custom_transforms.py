import torchvision.transforms as transforms

class PadToSquare():
    """
        This function takes PIL image as input and apply pad in order to obtain a squared image of dimensions 
        (max_dim, max_dim) where max_dim is the biggest between H or W
    """
    def __call__(self, img):
        H, W = img.size

        if H > W:
            return transforms.Pad(padding=[0, (H-W)//2])(img)

        if W > H:     
            return transforms.Pad(padding=[(W-H)//2, 0])(img)

        return img  

class DoNothing():
    """
        This function takes image and do nothing. It is useful for parametric callable transformation
    """
    def __call__(self, img):
        return img
