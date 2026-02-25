import torch
import matplotlib.pyplot as plt
import os

# Import the models and geometry utilities from your training script
from main import GeometricFlowNet, MetricDecoder, z_to_metric


def generate_images(num_images=4, num_steps=50, noise_scale=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    flow_net = GeometricFlowNet().to(device)
    decoder = MetricDecoder().to(device)

    if not os.path.exists("checkpoints/grfm_v2_flow.pth"):
        raise FileNotFoundError("Could not find flow model checkpoint. Train the model first!")

    flow_net.load_state_dict(torch.load("checkpoints/grfm_v2_flow.pth", map_location=device, weights_only=True))
    decoder.load_state_dict(torch.load("checkpoints/grfm_v2_decoder.pth", map_location=device, weights_only=True))

    flow_net.eval()
    decoder.eval()

    print(f"Running reverse Topological Flow Matching ({num_steps} steps)...")

    with torch.no_grad():
        # 2. Define the t=1 Starting State (Flat Geometry + Uniform Color)
        g_t = torch.zeros(num_images, 3, 32, 32, device=device)
        g_t[:, 0, :, :] = 1.0  # g11
        g_t[:, 1, :, :] = 1.0  # g22

        c_t = torch.full((num_images, 3, 32, 32), 0.5, device=device)

        # Inject "Micro-wrinkles" to ensure generation diversity
        g_t = g_t + torch.randn_like(g_t) * noise_scale
        c_t = c_t + torch.randn_like(c_t) * noise_scale

        # The flat target used for vector field calculations
        g_flat = torch.zeros_like(g_t)
        g_flat[:, 0, :, :] = 1.0
        g_flat[:, 1, :, :] = 1.0

        # 3. Euler Integration Loop (Reverse ODE)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            # Time flows backward from 1.0 down to 0.0
            t_val = 1.0 - (i * dt)
            t_tensor = torch.full((num_images, 1), t_val, device=device)

            # Predict the clean Data state (x_0)
            z_pred, c_pred = flow_net(g_t, c_t, t_tensor)
            g_pred = z_to_metric(z_pred)

            # Calculate the instantaneous Vector Field (dx/dt = x_noise - x_data)
            v_g = g_flat - g_pred
            v_c = 0.5 - c_pred

            # Euler Step backwards in time: x(t - dt) = x(t) - dt * v(t)
            g_t = g_t - dt * v_g
            c_t = c_t - dt * v_c

        # 4. The Final SPADE Rendering
        print("Rendering generated geometry into RGB...")
        final_rgb = decoder(c_t, g_t)

    # 5. Visualization
    fig, axes = plt.subplots(1, num_images, figsize=(4 * num_images, 4))
    if num_images == 1:
        axes = [axes]

    for i in range(num_images):
        img_to_show = final_rgb[i].cpu().permute(1, 2, 0).numpy()

        # Clip to [0, 1] just in case numerical errors push pixels out of bounds
        img_to_show = img_to_show.clip(0, 1)

        axes[i].imshow(img_to_show)
        axes[i].set_title(f"Generated {i + 1}")
        axes[i].axis('off')

    plt.suptitle("GRFM v2.0: Riemannian Geometry to RGB", fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    generate_images(num_images=4, num_steps=50, noise_scale=0.15)
