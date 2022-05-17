import requests
from PIL import Image

from model import get_model, get_result

def main():
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    model = get_model()
    print(f"model device cuda: {next(model.parameters()).is_cuda}")
    result = get_result(model, image)
    print(result)

if __name__ == '__main__':
    main()