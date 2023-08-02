from PIL import Image, ImageDraw, ImageFont

def generate_number_image(number, size, font_path, output_path,stroke_width):
    # Create a blank image with the specified size and white background
    image = Image.new('RGB', size, color='black')

    # Load the font
    font = ImageFont.truetype(font_path, size[1])

    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Calculate the position to center the number in the image
    text_width, text_height = draw.textsize(number, font=font)
    # x = (size[0] - text_width) // 2
    # y = (size[1] - text_height) // 2
    x=0
    y=0
    # Draw the number on the image
    draw.text((x, y), number, fill='white', font=font, stroke_width=stroke_width, stroke_fill='white')

    # Save the image to the output path
    image.save(output_path)
    


number = '16'
image_size = (140, 160)  # Width and height of the image in pixels
font_path = 'gas-cylinder-weight\code-testing\Kruti Dev 010 Regular.ttf'  # Replace with the path to your desired font file
output_path = 'gas-cylinder-weight\images/test_template/16.jpg'
stroke_width = 6

generate_number_image(number, image_size, font_path, output_path,stroke_width)





