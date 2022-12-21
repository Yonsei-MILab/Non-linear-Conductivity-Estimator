#Load network weight
model = EPT_Model().cpu()
#model.load_state_dict(torch.load("./model_weights.pth"))


#Load testing dataset
#dataset = io.loadmat('Data/testing_dataset.mat')
invivo_dataset = torch.tensor(dataset['dataset_training'])
invivo_dataset = np.expand_dims(invivo_dataset,1)
invivo_dataset = torch.tensor(invivo_dataset)

invivo_mask = torch.tensor(dataset['dataset_mask'])
invivo_mask = np.expand_dims(invivo_mask,1)
invivo_mask = torch.tensor(invivo_mask)

kernel_h, kernel_w = 11, 11
step, n_channels = 1, 1
invivo_dataset_patch = np.array(invivo_dataset.unfold(2, kernel_h, step).unfold(3, kernel_w, step).permute(2, 3, 0, 1, 4, 5).reshape(-1, n_channels, kernel_h, kernel_w))
invivo_mask_patch = np.array(invivo_mask.unfold(2, kernel_h, step).unfold(3, kernel_w, step).permute(2, 3, 0, 1, 4, 5).reshape(-1, n_channels, kernel_h, kernel_w))

invivo_dataset_patch_magnitude = torch.abs(invivo_dataset_patch)
invivo_dataset_patch_angle = torch.angle(invivo_dataset_patch)

[az,ch,ax,ay]=invivo_dataset_patch_angle.shape
for sslice in range(az):
    invivo_dataset_patch_angle_norm[sslice] = np.squeeze(invivo_dataset_patch_angle[sslice,:,:,:] - torch.min(invivo_dataset_patch_angle[sslice,:,:,:]))
invivo_dataset_patch_angle_norm = torch.tensor(invivo_dataset_patch_angle_norm)

invivo_dataset_patch_magnitude = invivo_dataset_patch_magnitude*invivo_mask_patch
invivo_dataset_patch_angle_norm = invivo_dataset_patch_angle_norm*invivo_mask_patch

invivo_dataset_norm = torch.cat((invivo_dataset_patch_magnitude, invivo_dataset_patch_angle_norm),1)

result = model(invivo_dataset_norm)
