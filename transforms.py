from torchvision import transforms

# transforms for labeled training and validation
tforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(size=32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ]),
    'HA': transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=.1, hue=.1),
        transforms.RandomAffine(degrees=(-10, 10), shear=(-15, 15), scale=(0.8, 1.2)),
        transforms.RandomCrop(size=32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ]),
}