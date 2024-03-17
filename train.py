from typing import Optional
from model.vae import Vae
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch
import torchvision.datasets as tvds
from dataclasses import dataclass
import torchvision.transforms as transforms
import time
import cv2
import os


@dataclass
class Trainer:
    in_channels: int
    latent_dim: int
    hidden_dims: Optional[list[int]]
    pretrain_num_epochs: int
    batch_size: int
    learning_rate: float
    datasets: str = "./datasets"
    checkpoint_path: str = "./checkpoints"
    img_dir: str = "./"

    def pretrain_CIFAR10(
        self,
    ):
        print(f"Is cuda available: { torch.cuda.is_available() }")
        print(f"How many devices: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Current device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        start = time.time()
        if self.pretrain_num_epochs == 0:
            with open(self.checkpoint_path + "/test", "w") as f:
                end = time.time() - start
                f.write("test")
                f.write(f"duration: {end}")

        print("Starting to pretrain")
        model = Vae(self.in_channels, self.latent_dim, self.hidden_dims)
        print("VAE model is loaded")
        train_dataset = tvds.CIFAR10(root=self.datasets, train=True, transform=transforms.ToTensor(), download=False)
        print("Train dataset is loaded")
        # test_dataset = tvds.CIFAR10(root=self.datasets, train=False, download=False)
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, shuffle=True
        )
        # test_loader = DataLoader(
        #     dataset=test_dataset, batch_size=self.batch_size, shuffle=False
        # )
        optimizer = Adam(model.parameters(), lr=self.learning_rate)
        # Training
        for epoch in range(self.pretrain_num_epochs):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.train()
            train_loss = 0
            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.to(device)
                optimizer.zero_grad()
                recon_batch, mu, log_var = model(data)
                loss = model.loss_function(recon_batch, data, mu, log_var)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()

            print(
                "Epoch [{}/{}], Loss: {:.4f}".format(
                    epoch + 1,
                    self.pretrain_num_epochs,
                    train_loss / (len(train_loader) * self.batch_size),
                )
            )
        end = time.time() - start
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "duration" : end,
                "loss": loss,
            },
            self.checkpoint_path + "/original",
        )
        print(f"duration: {end}")
        model.eval()
        return model

    @torch.no_grad()
    def make_n_img(self, model, num_im: int):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        samples = model.sample(num_im, "").to(device)
        os.mkdir("./images")
        for i in range(samples.size()[0]):
            img = samples[i, :, :, :].cpu().numpy()
            cv2.imwrite(f"./images/img{i}.png", img) # danger

def main():
    trainer = Trainer(in_channels=3, latent_dim=64, hidden_dims=None, pretrain_num_epochs=1000, batch_size=64, learning_rate=1e-4)
    model = trainer.pretrain_CIFAR10()
    trainer.make_n_img(model, 10)

if __name__ == "__main__":
    main()
