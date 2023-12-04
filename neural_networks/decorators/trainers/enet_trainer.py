from torch.nn.modules.loss import _Loss
import torch
from torch import nn
import tqdm
import numpy as np
from neural_networks.models.lanenet.enet import ENet
from data.datasets.lane_dataset import LaneDataset
from neural_networks.decorators.trainers.trainer import Trainer
from utils.config import ModelConfig


class DiscriminativeLoss(_Loss):
    def __init__(self, delta_var=0.5, delta_dist=3,
                 norm=2, alpha=1.0, beta=1.0, gamma=0.001,
                 device="cpu", reduction="mean", n_clusters=4):
        super(DiscriminativeLoss, self).__init__(reduction=reduction)
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.device = torch.device(device)
        self.n_clusters = n_clusters
        assert self.norm in [1, 2]

    def forward(self, input, target):
        assert not target.requires_grad

        return self._discriminative_loss(input, target)

    def _discriminative_loss(self, input, target):
        num_samples=target.size(0)

        dis_loss=torch.tensor(0.).to(self.device)
        var_loss=torch.tensor(0.).to(self.device)
        reg_loss=torch.tensor(0.).to(self.device)
        for i in range(num_samples):
            clusters=[]
            sample_embedding=input[i,:,:,:]
            sample_label=target[i,:,:].squeeze()
            num_clusters=len(sample_label.unique())-1
            vals=sample_label.unique()[1:]
            sample_label=sample_label.view(sample_label.size(0)*sample_label.size(1))
            sample_embedding=sample_embedding.view(-1,sample_embedding.size(1)*sample_embedding.size(2))
            v_loss=torch.tensor(0.).to(self.device)
            d_loss=torch.tensor(0.).to(self.device)
            r_loss=torch.tensor(0.).to(self.device)
            for j in range(num_clusters):
                indices=(sample_label==vals[j]).nonzero()
                indices=indices.squeeze()
                cluster_elements=torch.index_select(sample_embedding,1,indices)
                Nc=cluster_elements.size(1)
                mean_cluster=cluster_elements.mean(dim=1,keepdim=True)
                clusters.append(mean_cluster)
                v_loss+=torch.pow((torch.clamp(torch.norm(cluster_elements-mean_cluster)-self.delta_var,min=0.)),2).sum()/Nc
                r_loss+=torch.sum(torch.abs(mean_cluster))
            for index in range(num_clusters):
                for idx,cluster in enumerate(clusters):
                    if index==idx:
                        continue
                    else:
                        distance=torch.norm(clusters[index]-cluster)#torch.sqrt(torch.sum(torch.pow(clusters[index]-cluster,2)))
                        d_loss+=torch.pow(torch.clamp(self.delta_dist-distance,min=0.),2)
            var_loss+=v_loss/num_clusters
            dis_loss+=d_loss/(num_clusters*(num_clusters-1))
            reg_loss+=r_loss/num_clusters

        return self.alpha*(var_loss/num_samples)+self.beta*(dis_loss/num_samples)+self.gamma*(reg_loss/num_samples)


class EnetTrainer(Trainer):
    def __init__(self):
        dataset = LaneDataset()
        self._BATCH_SIZE = 8
        self._LR = 5e-4
        self._NUM_EPOCHS = ModelConfig().epochs

        model = ENet(2, 4)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=self._LR, weight_decay=0.0002)

        super().__init__(model, "enet_trainer", dataset, optimizer)
        self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self._model.to(self._device)

        self._train_dataloader = torch.utils.data.DataLoader(self._dataset, batch_size=self._BATCH_SIZE, shuffle=False)

    def train(self, save=True):
        for epoch in range(self._NUM_EPOCHS):
            self._model.train()
            losses = []
            for batch in tqdm.tqdm(self._train_dataloader):
                img, binary_target, instance_target = batch
                img = img.to(self._device)
                binary_target = binary_target.to(self._device)
                instance_target = instance_target.to(self._device)

                self._optimizer.zero_grad()

                binary_logits, instance_emb = self._model(img)

                binary_loss, instance_loss = self._compute_loss(binary_logits, instance_emb, binary_target, instance_target)
                loss = binary_loss + instance_loss
                loss.backward()

                self._optimizer.step()

                losses.append((binary_loss.detach().cpu(), instance_loss.detach().cpu()))

            mean_losses = np.array(losses).mean(axis=0)

            msg = (f"Epoch {epoch}:"
                   f"loss = {mean_losses}")
            print(msg)

            if save:
                self.save()

    def _compute_loss(self, binary_output, instance_output, binary_label, instance_label):
        ce_loss = nn.CrossEntropyLoss()
        binary_loss = ce_loss(binary_output, binary_label)

        ds_loss = DiscriminativeLoss(delta_var=0.5, delta_dist=3, alpha=1.0, beta=1.0, gamma=0.001, device="cuda")
        instance_loss = ds_loss(instance_output, instance_label)

        return binary_loss, instance_loss
