import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from utils import image_loader, imshow
import time

class NeuralStyleTransfer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn = models.vgg19(pretrained=True).features.to(self.device).eval()
        self.imsize = 512 if torch.cuda.is_available() else 128
        self.loader = transforms.Compose([
            transforms.Resize(self.imsize),
            transforms.ToTensor()
        ])
        
        # Normalization for VGG input
        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        
    def get_style_model_and_losses(self, content_img, style_img):
        """Build the style transfer model and compute losses"""
        content_layers = ['conv_4']
        style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        
        # Create a module to normalize input image
        normalization = Normalization(self.cnn_normalization_mean, self.cnn_normalization_std).to(self.device)
        
        content_losses = []
        style_losses = []
        
        model = nn.Sequential(normalization)
        
        i = 0  # increment every time we see a conv
        for layer in self.cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = f'conv_{i}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu_{i}'
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool_{i}'
            elif isinstance(layer, nn.BatchNorm2d):
                name = f'bn_{i}'
            else:
                raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')
            
            model.add_module(name, layer)
            
            if name in content_layers:
                # Add content loss
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module(f"content_loss_{i}", content_loss)
                content_losses.append(content_loss)
                
            if name in style_layers:
                # Add style loss
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module(f"style_loss_{i}", style_loss)
                style_losses.append(style_loss)
        
        # Trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break
        
        model = model[:(i + 1)]
        
        return model, style_losses, content_losses
    
    def run_style_transfer(self, content_img, style_img, input_img, num_steps=300,
                         style_weight=1000000, content_weight=1):
        """Run the style transfer"""
        print('Building the style transfer model...')
        model, style_losses, content_losses = self.get_style_model_and_losses(
            content_img, style_img
        )
        
        # Use the input image as the starting point
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        
        print('Optimizing...')
        run = [0]
        start_time = time.time()
        
        while run[0] <= num_steps:
            def closure():
                # Correct the values of updated input image
                input_img.data.clamp_(0, 1)
                
                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0
                
                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss
                
                style_score *= style_weight
                content_score *= content_weight
                
                loss = style_score + content_score
                loss.backward()
                
                run[0] += 1
                if run[0] % 50 == 0:
                    print(f"run {run[0]}:")
                    print(f"Style Loss: {style_score.item():.4f} "
                          f"Content Loss: {content_score.item():.4f}")
                    print(f"Time elapsed: {time.time() - start_time:.2f}s")
                    print()
                
                return style_score + content_score
            
            optimizer.step(closure)
        
        # Clamp the final image
        input_img.data.clamp_(0, 1)
        
        return input_img

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
    
    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()
    
    def forward(self, input):
        G = self.gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input
    
    @staticmethod
    def gram_matrix(input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
    
    def forward(self, img):
        return (img - self.mean) / self.std

def main():
    print("CODTECH NEURAL STYLE TRANSFER")
    print("----------------------------\n")
    
    nst = NeuralStyleTransfer()
    
    while True:
        print("\nOptions:")
        print("1. Apply style transfer")
        print("2. Exit")
        
        choice = input("Select an option (1-2): ")
        
        if choice == '2':
            print("Exiting style transfer application...")
            break
            
        if choice == '1':
            content_path = input("Enter path to content image: ")
            style_path = input("Enter path to style image: ")
            output_path = input("Enter path to save result (leave blank to show only): ")
            
            try:
                content_img = image_loader(content_path, nst.loader, nst.device)
                style_img = image_loader(style_path, nst.loader, nst.device)
                
                # Start with the content image or white noise
                input_img = content_img.clone()
                # input_img = torch.randn(content_img.data.size(), device=device)
                
                # Run style transfer
                output = nst.run_style_transfer(
                    content_img, style_img, input_img
                )
                
                # Display and save result
                plt.figure()
                imshow(output, title='Output Image')
                
                if output_path:
                    output_image = transforms.ToPILImage()(output.cpu().squeeze(0))
                    output_image.save(output_path)
                    print(f"Result saved to {output_path}")
                
                plt.show()
            except Exception as e:
                print(f"Error: {str(e)}")
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
