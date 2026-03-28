from PIL import Image
from torchvision import transforms
from model import Model
import torch
import cv2

cam = cv2.VideoCapture(0)
cv2.namedWindow("test")

model = Model().to('cuda')
model.load_state_dict(torch.load("cifar10_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((320,320)),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
])

predicted = 0
classes = ["closed", "open"]


while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    img_name = f"test.png"
    cv2.imwrite(img_name, frame)

    img = Image.open("test.png")
    img = transform(img).unsqueeze(0).to('cuda')

    model.eval()

    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, dim=1)

    if predicted != classes[pred.item()]:
        predicted = classes[pred.item()]
        print("your hand is", predicted)

    
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break