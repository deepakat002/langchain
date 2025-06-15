from langchain.text_splitter import RecursiveCharacterTextSplitter,Language

text = """
class ImageHandler:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = None

    def read_image(self):
        self.image = cv2.imread(self.image_path)
        if self.image is not None:
            print("Image loaded successfully.")
        else:
            print("Failed to load image.")

    def save_image(self, save_path):
        if self.image is not None:
            cv2.imwrite(save_path, self.image)
            print(f"Image saved at {save_path}.")
        else:
            print("No image to save. Please read an image first.")

# Example usage
handler = ImageHandler("dog.jpg")
handler.read_image()
handler.save_image("saved_dog.jpg")

"""

# Initialize the splitter
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=294,
    chunk_overlap=0,
)

# Perform the split
chunks = splitter.split_text(text)
print(chunks)

print("\n\n -----------------  1  ---------------------  \n\n")
print(len(chunks))
print(chunks[0])