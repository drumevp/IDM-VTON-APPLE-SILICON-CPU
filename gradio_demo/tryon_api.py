import sys
import os
import io
import logging
import tempfile
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError
import torch
from tryon_cli import initialize_pipeline, start_tryon
import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools

# ------------------------------
# 1. Logging Configuration
# ------------------------------

logging.basicConfig(
    level=logging.ERROR,  # Set to DEBUG for detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("tryon_api")

# ------------------------------
# 2. FastAPI Initialization
# ------------------------------

app = FastAPI(
    title="IDM-VITON API",
    description="API for Virtual Try-On using IDM-VITON",
    version="1.0"
)

# ------------------------------
# 3. Pipeline Initialization
# ------------------------------

try:
    pipe, openpose_model, parsing_model, tensor_transform = initialize_pipeline()
    logger.info("Pipeline initialized successfully.")
except Exception as e:
    logger.exception("Failed to initialize the pipeline.")
    raise e  # Terminate if pipeline initialization fails

# ------------------------------
# 4. ThreadPoolExecutor Setup
# ------------------------------

executor = ThreadPoolExecutor(max_workers=4)  # Adjust based on server capacity

# ------------------------------
# 5. Helper Function to Run Blocking Code
# ------------------------------

async def run_tryon_in_thread(*args, **kwargs):
    """
    Runs the blocking `start_tryon` function in a separate thread.
    """
    func = functools.partial(start_tryon, **kwargs)
    loop = asyncio.get_running_loop()
    logger.debug(f"Submitting start_tryon to executor with kwargs: {kwargs}")
    await loop.run_in_executor(executor, func)
    logger.debug("start_tryon completed.")

# ------------------------------
# 6. Try-On Endpoint
# ------------------------------

@app.post("/tryon/")
async def tryon_endpoint(
    human_image: UploadFile = File(..., description="Human image file (PNG or JPEG)"),
    garment_image: UploadFile = File(..., description="Garment image file (PNG or JPEG)"),
    garment_description: str = Form("", description="Description of the garment (optional)"),
    use_auto_mask: bool = Form(False, description="Use auto-generated mask"),
    use_auto_crop: bool = Form(False, description="Use auto-crop and resizing"),
    denoise_steps: int = Form(20, description="Number of denoising steps"),
    seed: int = Form(42, description="Random seed for reproducibility"),
):
    try:
        logger.info("Received a try-on request.")

        # ------------------------------
        # 6.1. Validate Uploaded Files
        # ------------------------------

        if human_image.content_type not in ["image/png", "image/jpeg"]:
            logger.error(f"Invalid human image format: {human_image.content_type}")
            raise HTTPException(status_code=400, detail="Invalid human image format. Only PNG and JPEG are supported.")
        if garment_image.content_type not in ["image/png", "image/jpeg"]:
            logger.error(f"Invalid garment image format: {garment_image.content_type}")
            raise HTTPException(status_code=400, detail="Invalid garment image format. Only PNG and JPEG are supported.")

        # ------------------------------
        # 6.2. Read and Save Images
        # ------------------------------

        human_image_bytes = await human_image.read()
        garment_image_bytes = await garment_image.read()

        # Use TemporaryDirectory to handle temp files safely
        with tempfile.TemporaryDirectory() as tmpdirname:
            human_img_path = os.path.join(tmpdirname, "human_image.png")
            garment_img_path = os.path.join(tmpdirname, "garment_image.png")
            output_path = os.path.join(tmpdirname, "output_tryon.png")

            # Save uploaded images to temp files
            logger.debug(f"Saving human image to {human_img_path}")
            with open(human_img_path, "wb") as f:
                f.write(human_image_bytes)
            logger.debug(f"Saving garment image to {garment_img_path}")
            with open(garment_img_path, "wb") as f:
                f.write(garment_image_bytes)

            logger.info("Running try-on process.")

            # ------------------------------
            # 6.3. Define Try-On Parameters
            # ------------------------------

            tryon_kwargs = {
                'pipe': pipe,
                'openpose_model': openpose_model,
                'parsing_model': parsing_model,
                'tensor_transform': tensor_transform,
                'human_image_path': human_img_path,
                'garment_image_path': garment_img_path,
                'garment_description': garment_description,
                'use_auto_mask': use_auto_mask,
                'use_auto_crop': use_auto_crop,
                'denoise_steps': denoise_steps,
                'seed': seed,
                'output_path': output_path
            }

            # ------------------------------
            # 6.4. Create Try-On Task
            # ------------------------------

            tryon_task = asyncio.create_task(run_tryon_in_thread(**tryon_kwargs))

            # ------------------------------
            # 6.5. Await Try-On Task Completion
            # ------------------------------

            await tryon_task

            # ------------------------------
            # 6.6. Check Try-On Task Result
            # ------------------------------

            # Since exceptions in run_tryon_in_thread will propagate, no need for additional checks here

            # ------------------------------
            # 6.7. Read and Return Output Image
            # ------------------------------

            if not os.path.exists(output_path):
                logger.error("Output image not found.")
                raise HTTPException(status_code=500, detail="Try-on process failed to generate output image.")

            logger.debug(f"Reading output image from {output_path}")
            with open(output_path, "rb") as f:
                output_image = f.read()

            logger.info("Try-on process completed successfully.")
            logger.info("Returning the output image.")

            return StreamingResponse(io.BytesIO(output_image), media_type="image/png")

    except HTTPException as he:
        # Re-raise HTTP exceptions
        logger.error(f"HTTPException: {he.detail}")
        raise he
    except Exception as e:
        # Log the exception with stack trace
        logger.exception("An error occurred during the try-on process.")
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again later.")
