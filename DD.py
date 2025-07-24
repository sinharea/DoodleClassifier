from neuralNetwork import NeuralNetwork
from random import shuffle, sample
from os import listdir
import numpy as np
import pygame
import asyncio
import platform
import matplotlib.pyplot as plt

# ------------------------------ CONSTANTS ------------------------------
DATA_FOLDER = "data"
PREVIEW_SIZE = 28
CANVAS_SIZE = 560  # 28x28 grid scaled by 20
PIXEL_SIZE = 20
WINDOW_WIDTH = CANVAS_SIZE
WINDOW_HEIGHT = CANVAS_SIZE + 50
FPS = 60

# ------------------------------ LOAD DATA ------------------------------
def load_and_prepare_data():
    global label_of_categories, all_of_the_doodles, training_set, test_set, target
    files = [f for f in listdir(DATA_FOLDER) if f.endswith(".npy")]
    label_of_categories = [f[:-4] for f in files]
    all_of_the_doodles = [np.load(f"{DATA_FOLDER}/{f}") for f in files]

    target = {
        label: [1 if i == idx else 0 for i in range(len(files))]
        for idx, label in enumerate(label_of_categories)
    }

    data_set = []
    for i, doodles in enumerate(all_of_the_doodles):
        num_samples = min(len(doodles), 1000)
        print(f"Category {label_of_categories[i]}: {num_samples} samples")
        for doodle in doodles[:num_samples]:
            data_set.append((label_of_categories[i], doodle.flatten() / 255.0))

    shuffle(data_set)
    training_size = int(len(data_set) * 0.9)
    training_set = data_set[:training_size]
    test_set = data_set[training_size:]
    print(f"Training samples: {len(training_set)}")
    print(f"Test samples: {len(test_set)}")
    for label, data in test_set[:5]:
        print(f"Sample {label} min: {data.min()}, max: {data.max()}")

# ------------------------------ TRAIN & TEST ------------------------------
def train_model(epochs=100, learning_rate=0.001):
    nn.learning_rate = learning_rate
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        shuffle(training_set)
        for label, data in training_set:
            nn.train(data, target[label])
        correct = sum(1 for label, data in test_set if label_of_categories[nn.predict(data).index(max(nn.predict(data)))] == label)
        print(f"Validation Accuracy: {correct / len(test_set) * 100:.2f}%")

def test_model():
    correct = 0
    random_samples = sample(test_set, min(5, len(test_set)))
    for i, (label, data) in enumerate(random_samples):
        prediction = nn.predict(data)
        predicted = label_of_categories[prediction.index(max(prediction))]
        plt.figure(figsize=(4, 4))
        plt.imshow(data.reshape(28, 28) * 255.0, cmap='gray', vmin=0, vmax=255)
        plt.title(f"Sample {i+1}: True={label}, Predicted={predicted}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        print(f"Sample {i+1}: True label: {label}, Predicted: {predicted}")
        print("Prediction probabilities:")
        for cat, prob in zip(label_of_categories, prediction):
            print(f"  {cat}: {prob:.6f}")
        print()

    for label, data in test_set:
        prediction = nn.predict(data)
        predicted = label_of_categories[prediction.index(max(prediction))]
        if predicted == label:
            correct += 1
    accuracy = correct / len(test_set) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

# ------------------------------ PLAY FUNCTION ------------------------------
def play():
    global nn, label_of_categories, test_set
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Doodle Prediction")
    font = pygame.font.SysFont("arial", 24)

    print(f"Categories: {label_of_categories}")
    print(f"Number of output nodes: {nn.number_of_nodes_in_each_layer[-1]}")

    cup_samples = [data for label, data in test_set if label == "cup"]
    if cup_samples:
        cup_image = cup_samples[0].reshape(28, 28) * 255.0
        plt.figure(figsize=(8, 8))
        plt.imshow(cup_image, cmap='gray', vmin=0, vmax=255)
        plt.title("Sample Cup from Test Set")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    drawing = np.zeros((28, 28), dtype=np.float32)
    print(f"Initial drawing array shape: {drawing.shape}")
    clock = pygame.time.Clock()
    running = True
    prediction = None

    def draw_pixel(x, y, value):
        grid_x, grid_y = x // PIXEL_SIZE, y // PIXEL_SIZE
        if 0 <= grid_x < 28 and 0 <= grid_y < 28:
            drawing[grid_y, grid_x] = value
            color = (255 * value, 255 * value, 255 * value)
            pygame.draw.rect(screen, color, (grid_x * PIXEL_SIZE, grid_y * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE))

    def clear_canvas():
        nonlocal prediction
        drawing.fill(0)
        print(f"Drawing array shape after clear: {drawing.shape}")
        screen.fill((0, 0, 0), (0, 0, CANVAS_SIZE, CANVAS_SIZE))
        prediction = None

    def get_prediction():
        input_data = drawing.flatten()  # Remove / 255.0 to match training data
        print(f"Input data shape for prediction: {input_data.shape}")
        print(f"Input data min: {input_data.min()}, max: {input_data.max()}")
        prediction = nn.predict(input_data)
        print("Prediction probabilities:")
        for label, prob in zip(label_of_categories, prediction):
            print(f"  {label}: {prob:.6f}")
        print(f"Predicted index: {prediction.index(max(prediction))}")
        return prediction

    def plot_input_image():
        print(f"Drawing array shape before plotting: {drawing.shape}")
        plt.figure(figsize=(8, 8))
        plt.imshow(drawing, cmap='gray', vmin=0, vmax=1)
        plt.title("Input Doodle")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    async def main_loop():
        nonlocal running, prediction
        clear_canvas()
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.pos[1] < CANVAS_SIZE:
                    if event.button == 1:
                        draw_pixel(event.pos[0], event.pos[1], 1.0)
                    elif event.button == 3:
                        draw_pixel(event.pos[0], event.pos[1], 0.0)
                elif event.type == pygame.MOUSEMOTION and pygame.mouse.get_pressed()[0] and event.pos[1] < CANVAS_SIZE:
                    draw_pixel(event.pos[0], event.pos[1], 1.0)
                elif event.type == pygame.MOUSEBUTTONDOWN and CANVAS_SIZE <= event.pos[1] < WINDOW_HEIGHT:
                    if event.button == 1:
                        submit_rect = pygame.Rect(10, CANVAS_SIZE + 10, 80, 30)
                        if submit_rect.collidepoint(event.pos):
                            prediction = get_prediction()
                            print(f"Predicted category: {label_of_categories[prediction.index(max(prediction))]}")
                            plot_input_image()
                        clear_rect = pygame.Rect(100, CANVAS_SIZE + 10, 80, 30)
                        if clear_rect.collidepoint(event.pos):
                            clear_canvas()

            screen.fill((100, 100, 100), (0, CANVAS_SIZE, CANVAS_SIZE, 50))
            submit_text = font.render("Submit", True, (255, 255, 255))
            screen.blit(submit_text, (10, CANVAS_SIZE + 10))
            clear_text = font.render("Clear", True, (255, 255, 255))
            screen.blit(clear_text, (100, CANVAS_SIZE + 10))
            pred_text = font.render(f"Prediction: {label_of_categories[prediction.index(max(prediction))]}" if prediction else "Prediction: None", True, (255, 255, 255))
            screen.blit(pred_text, (190, CANVAS_SIZE + 10))

            pygame.display.flip()
            clock.tick(FPS)
            await asyncio.sleep(1.0 / FPS)

    if platform.system() != "Emscripten":
        asyncio.run(main_loop())
    else:
        asyncio.ensure_future(main_loop())

    pygame.quit()

if __name__ == "__main__":
    load_and_prepare_data()
    try:
        nn = NeuralNetwork.load("model.pkl")
        print("Model loaded from model.pkl")
    except Exception as e:
        print(f"Failed to load model: {e}. Training new model.")
        nn = NeuralNetwork(784, 128, 64, len(label_of_categories))
        train_model(epochs=100, learning_rate=0.001)
        nn.save("model.pkl")
        print("Model trained and saved")
    test_model()
    play()