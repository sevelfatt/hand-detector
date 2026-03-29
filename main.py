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

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(img_rgb)
    img = transform(img).unsqueeze(0).to('cuda')

    model.eval()

    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, dim=1)

    cv2.putText(
        frame,
        f"Your hand is: {classes[pred.item()]}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("test", frame)
    
    if predicted != classes[pred.item()]:
        predicted = classes[pred.item()]
        print("your hand is", predicted)

    
    k = cv2.waitKey(1)
    if k%256 == 27:
        print("Escape hit, closing...")
        break