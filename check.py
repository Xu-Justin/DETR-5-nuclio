import requests
from PIL import Image

from model import get_model, get_result

def main():
    image = Image.open('resources/sample.jpg')
    model = get_model()
    print(f"model device cuda: {next(model.parameters()).is_cuda}")
    result = get_result(model, image)
    print(result)

if __name__ == '__main__':
    main()