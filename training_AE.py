import torch
import torch.optim as optim
import torch.nn.functional as F
from architecture_2 import AutoEncoder
from dataset_loader import AEDataset
import os

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    num_batch = 0   
    for batch_idx, sample in enumerate(train_loader):
        num_batch += 1
        
        data = sample.to(device)   
        optimizer.zero_grad()
        recon_batch = model(data)
        # loss = F.mse_loss(data, recon_batch)
        # loss = chamfer_distance(torch.swapaxes(data, 1, 2), torch.swapaxes(recon_batch, 1, 2))
        # loss = model.get_loss(data, recon_batch)
        loss = model.get_loss(data.permute(0,2,1), recon_batch.permute(0,2,1))
        # loss = F.mse_loss(data.permute(0,2,1), recon_batch.permute(0,2,1))
        # loss = F.mse_loss(data[:,0,:].squeeze(), recon_batch[:,0,:].squeeze()) + F.mse_loss(data[:,1,:].squeeze(), recon_batch[:,1,:].squeeze()) + F.mse_loss(data[:,2,:].squeeze(), recon_batch[:,2,:].squeeze())
        # import pdb; pdb.set_trace()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(sample), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    print('====> Epoch: {} Average loss: {:.6f}'.format(
              epoch, train_loss/num_batch))  

def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    num_batch = 0  
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            num_batch += 1
            data = sample.to(device)
            recon_batch = model(data)

            # test_loss += F.mse_loss(data, recon_batch)
            # test_loss += model.get_loss(data, recon_batch).item()
            test_loss += model.get_loss(data.permute(0,2,1), recon_batch.permute(0,2,1)).item()
            # test_loss += F.mse_loss(data.permute(0,2,1), recon_batch.permute(0,2,1))

    test_loss /= num_batch

    print('Test set: Average loss: {:.10f}\n'.format(test_loss))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)

if __name__ == "__main__":
    torch.manual_seed(2021)
    device = torch.device("cuda")

    train_len = 9500
    test_len = 500
    total_len = train_len + test_len

    dataset = AEDataset(percentage = 1.0)
    train_dataset = torch.utils.data.Subset(dataset, range(0, train_len))
    test_dataset = torch.utils.data.Subset(dataset, range(train_len, total_len))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=True)

    
    print("training data: ", len(train_dataset))
    print("test data: ", len(test_dataset))
    print("data path:", dataset.dataset_path)



    model = AutoEncoder(normal_channel=False).to(device)
    model.apply(weights_init)

    weight_path = "/home/baothach/shape_servo_data/teleoperation/sanity_check_examples/ex_2/autoencoder/weights"
    # model.load_state_dict(torch.load(os.path.join(weight_path, "epoch " + str(30))))

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.1)
    for epoch in range(0, 151):
        train(model, device, train_loader, optimizer, epoch)
        scheduler.step()
        test(model, device, test_loader, epoch)
        
        if epoch % 1 == 0:            
            torch.save(model.state_dict(), os.path.join(weight_path, "epoch " + str(epoch)))

