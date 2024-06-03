# Step 1: Install Necessary Libraries
!pip install transformers
!pip install ipywidgets

# Step 2: Import Libraries and Set Up the Chatbot
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Load the pre-trained BlenderBot model and tokenizer
model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

# Step 3: Create a Function to Interact with the Chatbot
def get_response(user_input):
    inputs = tokenizer(user_input, return_tensors="pt")
    reply_ids = model.generate(**inputs)
    response = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    return response

# Step 4: Create a User Interface with IPython Widgets
from IPython.display import display
import ipywidgets as widgets

# Create a text input widget
input_box = widgets.Text(
    value='',
    placeholder='Type your message here',
    description='You:',
    disabled=False
)

# Create a submit button widget
submit_button = widgets.Button(
    description='Submit',
    disabled=False,
    button_style='',
    tooltip='Click to submit your message',
    icon='check'
)

# Create an output widget
output_box = widgets.Output()

def on_submit(button):
    user_input = input_box.value
    if user_input:
        with output_box:
            print(f"You: {user_input}")
            response = get_response(user_input)
            print(f"Bot: {response}")
        input_box.value = ''  # Clear the input box after submission

# Set the function to call when the button is clicked
submit_button.on_click(on_submit)

# Display the widgets
display(input_box)
display(submit_button)
display(output_box)
