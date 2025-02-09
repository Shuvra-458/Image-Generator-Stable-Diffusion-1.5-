import requests
import time
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# ======================== STEP 1: OPTIMIZE PROMPT ======================== #

def optimize_prompt(prompt):
    """Uses GPT-2 to refine and optimize the input prompt"""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Set pad_token_id to eos_token_id to avoid warnings
    tokenizer.pad_token = tokenizer.eos_token

    # Encode input with attention_mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    outputs = model.generate(
        inputs["input_ids"], 
        attention_mask=inputs["attention_mask"],  
        max_length=30,
        num_return_sequences=1,  
        no_repeat_ngram_size=2,  
        temperature=0.7,
        top_p=0.9,
        top_k=40
    )

    optimized_prompt = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Trim to the first sentence
    if "." in optimized_prompt:
        optimized_prompt = optimized_prompt.split(".")[0]  

    return optimized_prompt

# ======================== STEP 2: GENERATE IMAGE USING COMFYUI ======================== #
def generate_image(optimized_prompt):
    """Sends the optimized prompt to ComfyUI for image generation"""
    url = 'http://127.0.0.1:8188/prompt'

    payload = {
    "prompt": {
        "4": {  # Load Checkpoint (Keep it as 4 if ComfyUI shows this number)
            "inputs": {
                "ckpt_name": "v1-5-pruned-emaonly.safetensors"
            },
            "class_type": "Load Checkpoint"
        },
        "6": {  
            "inputs": {
                "clip": [4, "CLIP"],  # Reference Load Checkpoint
                "text": "beautiful scenery nature glass bottle landscape, purple galaxy bottle"
            },
            "class_type": "CLIP Text Encode"
        },
        "7": {  
            "inputs": {
                "clip": [4, "CLIP"],  # Reference Load Checkpoint
                "text": "text, watermark"
            },
            "class_type": "CLIP Text Encode"
        },
        "5": {  
            "inputs": {
                "width": 512, 
                "height": 512, 
                "batch_size": 1
            },
            "class_type": "Empty Latent Image"
        },
        "3": {  
            "inputs": {
                "model": [4, "MODEL"],  # Reference Load Checkpoint
                "positive": [6, "CONDITIONING"],
                "negative": [7, "CONDITIONING"],
                "latent_image": [5, "LATENT"],
                "seed": 15680208700286,
                "steps": 20,
                "cfg": 8.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0
            },
            "class_type": "KSampler"
        },
        "8": {  
            "inputs": {
                "samples": [3, "LATENT"], 
                "vae": [4, "VAE"]  # Reference Load Checkpoint
            },
            "class_type": "VAE Decode"
        },
        "9": {  
            "inputs": {
                "images": [8, "IMAGE"], 
                "filename_prefix": "ComfyUI"
            },
            "class_type": "Save Image"
        }
    }
}








    response = requests.post(url, json=payload)

    if response.status_code != 200:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return None

    try:
        response_json = response.json()
        print("🔍 API Response:", response_json)  # Debugging line

        if "error" in response_json:
            print("❌ API Error:", response_json["error"]["message"])
            return None

        return response_json  # Return full response for further processing

    except Exception as e:
        print("❌ Error decoding JSON response:", e)
        return None
# ======================== STEP 3: WAIT FOR IMAGE & DOWNLOAD ======================== #

def wait_for_image():
    """Waits for the image to be generated and downloads it"""
    job_status_url = 'http://127.0.0.1:8188/history'

    while True:
        time.sleep(2)  # Wait for 2 seconds before checking again
        status_response = requests.get(job_status_url)
        
        if status_response.status_code == 200:
            job_data = status_response.json()
            if job_data and "outputs" in job_data[-1]:  # Check if output is ready
                image_url = job_data[-1]["outputs"][0]["filename"]
                return download_image(image_url)
        
        print("⏳ Still processing...")

def download_image(image_url):
    """Downloads the generated image"""
    img_response = requests.get(f"http://127.0.0.1:8188/{image_url}")
    
    if img_response.status_code == 200:
        with open("generated_image.png", "wb") as f:
            f.write(img_response.content)
        print("✅ Image saved as generated_image.png")
    else:
        print(f"❌ Error downloading image: {img_response.status_code}")

# ======================== MAIN EXECUTION ======================== #

if __name__ == "__main__":
    user_prompt = input("Enter your image description: ")  # Example: "A beautiful futuristic city at sunset"
    
    print("\n🔍 Optimizing prompt using GPT-2...")
    optimized_prompt = optimize_prompt(user_prompt)
    print(f"✨ Optimized Prompt: {optimized_prompt}\n")

    generate_image(optimized_prompt)
