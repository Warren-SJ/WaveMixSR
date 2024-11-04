import torch
import torchvision
import numpy as np
from PIL import Image
import kornia
from wavemixmodel import WaveMixSR
import gradio as gr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_target = torchvision.transforms.Compose(
        [   torchvision.transforms.ToTensor(),
     ])

weights = torch.load('weights.pth', map_location=device)
model = WaveMixSR(depth = 4, mult = 1, ff_channel = 144, final_dim = 144, dropout = 0.3, scale_factor = 2).to(device)
model.load_state_dict(weights)
model.eval()

def process_image(input_image):
    img = Image.fromarray(input_image)
    img = transform_target(img)
    
    # Ensure the image has 3 channels
    if img.shape[0] > 3:
        img = img[:3, :, :]
    
    img = kornia.color.rgb_to_ycbcr(img)
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        output = kornia.color.ycbcr_to_rgb(output)
        output = output.squeeze(0)
        output = output.to('cpu')
        output = output.detach().numpy()
        output = output.transpose(1, 2, 0)
        output = np.clip(output, 0, 1)
        output = (output * 255).astype(np.uint8)
        output_image = Image.fromarray(output)
        return output_image

# Create a Gradio interface
iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Image(type="numpy"),
    title="Image Super-Resolution with WaveMix",
    description="Upload an image to enhance its resolution using the WaveMix model."
)

iface.launch()