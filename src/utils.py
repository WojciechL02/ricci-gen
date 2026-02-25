import os
import torch
import torch.nn.functional as F
from torchvision.utils import save_image


def compute_spatial_derivatives(tensor):
    """Computes spatial gradients (dx, dy) using finite differences."""
    dx = tensor[:, :, :, 1:] - tensor[:, :, :, :-1]
    dy = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return dx, dy


def image_to_metric(image, lambda_scale=1.0):
    """Calculates ground truth induced metric g_data from an RGB image."""
    dx, dy = compute_spatial_derivatives(image)
    g11 = 1.0 + lambda_scale * torch.sum(dx * dx, dim=1, keepdim=True)
    g22 = 1.0 + lambda_scale * torch.sum(dy * dy, dim=1, keepdim=True)
    g12 = lambda_scale * torch.sum(dx * dy, dim=1, keepdim=True)
    return torch.cat([g11, g22, g12], dim=1)  # (B, 3, H, W)


def z_to_metric(z, lambda_scale=1.0):
    """
    Analytically calculates a guaranteed Gauss-Codazzi valid metric
    from a 3D embedding space (z1, z2, z3), fixing the dimensionality bottleneck.
    """
    dz_dx, dz_dy = compute_spatial_derivatives(z)
    g11 = 1.0 + lambda_scale * torch.sum(dz_dx * dz_dx, dim=1, keepdim=True)
    g22 = 1.0 + lambda_scale * torch.sum(dz_dy * dz_dy, dim=1, keepdim=True)
    g12 = lambda_scale * torch.sum(dz_dx * dz_dy, dim=1, keepdim=True)
    return torch.cat([g11, g22, g12], dim=1)


def compute_color_anchor(x):
    """Heavy blur to extract low-frequency color, destroying geometry."""
    kernel = torch.ones(3, 1, 7, 7, device=x.device) / 49.0
    return F.conv2d(x, kernel, padding=3, groups=3)


def get_grad_norm(model):
    """Calculates the L2 norm of the gradients for a given model."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


@torch.no_grad()
def generate_and_log_images(flow_net, decoder, writer, epoch, device, num_images=8, num_steps=25, noise_scale=0.1):
    """Runs the reverse ODE to generate images, logs to TensorBoard, and saves to disk."""
    flow_net.eval()
    decoder.eval()

    os.makedirs("samples", exist_ok=True)
    # t=1 starting state
    g_t = torch.zeros(num_images, 3, 32, 32, device=device)
    g_t[:, 0, :, :] = 1.0  # g11
    g_t[:, 1, :, :] = 1.0  # g22
    c_t = torch.full((num_images, 3, 32, 32), 0.5, device=device)

    # Inject initial entropy (micro-wrinkles)
    g_t = g_t + torch.randn_like(g_t) * noise_scale
    c_t = c_t + torch.randn_like(c_t) * noise_scale

    g_flat = torch.zeros_like(g_t)
    g_flat[:, 0, :, :] = 1.0
    g_flat[:, 1, :, :] = 1.0

    dt = 1.0 / num_steps
    for i in range(num_steps):
        t_val = 1.0 - (i * dt)
        t_tensor = torch.full((num_images, 1), t_val, device=device)

        z_pred, c_pred = flow_net(g_t, c_t, t_tensor)
        g_pred = z_to_metric(z_pred)

        # Instantaneous vector field prediction
        v_g = g_flat - g_pred
        v_c = 0.5 - c_pred

        # Euler Step
        g_t = g_t - dt * v_g
        c_t = c_t - dt * v_c

    # Render final RGB and clip to valid image bounds [0, 1]
    final_rgb = decoder(c_t, g_t).clip(0, 1)

    # 1. Log to TensorBoard
    writer.add_images('Generation/Epoch_Samples', final_rgb, epoch)

    # 2. Save locally to disk as a PNG grid
    # nrow=4 means it will organize the 8 images into 2 rows of 4
    file_path = f"samples/epoch_{epoch:03d}.png"
    save_image(final_rgb, file_path, nrow=4)
    flow_net.train()
    decoder.train()


def train(flow_net, decoder, opt_flow, opt_decoder, batch_size, device, epochs, loader, writer):
    g_flat_template = torch.zeros(batch_size, 3, 32, 32, device=device)
    g_flat_template[:, 0, :, :] = 1.0  # g11
    g_flat_template[:, 1, :, :] = 1.0  # g22

    print("Starting GRFM v2.0 Joint Training...")
    global_step = 0
    for epoch in range(epochs):
        for batch_idx, (images, _) in enumerate(loader):
            images = images.to(device)

            # --- 1. PREPARE GROUND TRUTH ---
            g_data = image_to_metric(images)
            c_data = compute_color_anchor(images)

            # --- 2. FORWARD PROCESS (FLOW MATCHING) ---
            t = torch.rand(batch_size, 1, device=device)
            t_spatial = t.view(batch_size, 1, 1, 1)
            g_t = (1 - t_spatial) * g_data + t_spatial * g_flat_template
            c_t = (1 - t_spatial) * c_data + t_spatial * 0.5

            # --- 3. TRAIN FLOW NET ---
            opt_flow.zero_grad()
            z_pred, c_pred = flow_net(g_t, c_t, t)
            g_pred = z_to_metric(z_pred)

            loss_metric = F.mse_loss(g_pred, g_data)
            loss_color = F.mse_loss(c_pred, c_data)
            loss_flow = loss_metric + loss_color
            loss_flow.backward()
            grad_norm_flow = get_grad_norm(flow_net)  # Capture gradient norm
            opt_flow.step()

            # --- 4. TRAIN SPADE DECODER ---
            opt_decoder.zero_grad()

            rgb_pred = decoder(c_data, g_data)
            loss_decode = F.mse_loss(rgb_pred, images)
            loss_decode.backward()
            grad_norm_decoder = get_grad_norm(decoder)  # Capture gradient norm
            opt_decoder.step()

            if global_step % 10 == 0:
                writer.add_scalar('Loss/Flow_Total', loss_flow.item(), global_step)
                writer.add_scalar('Loss/Flow_Metric', loss_metric.item(), global_step)
                writer.add_scalar('Loss/Flow_Color', loss_color.item(), global_step)
                writer.add_scalar('Loss/Decoder', loss_decode.item(), global_step)
                writer.add_scalar('Gradients/Flow_Net_Norm', grad_norm_flow, global_step)
                writer.add_scalar('Gradients/Decoder_Norm', grad_norm_decoder, global_step)

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch}/{epochs}] Batch [{batch_idx}/{len(loader)}] "
                      f"| Flow Loss: {loss_flow.item():.4f} | Decode Loss: {loss_decode.item():.4f}")

            global_step += 1
            if batch_idx == 100:
                break

        print(f"Epoch {epoch} complete. Generating test images for TensorBoard...")
        generate_and_log_images(flow_net, decoder, writer, epoch, device, num_steps=25)

        os.makedirs("checkpoints", exist_ok=True)
        torch.save(flow_net.state_dict(), "checkpoints/grfm_v2_flow.pth")
        torch.save(decoder.state_dict(), "checkpoints/grfm_v2_decoder.pth")

    print("Training Complete!")
