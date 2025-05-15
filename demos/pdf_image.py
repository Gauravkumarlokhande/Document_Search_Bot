import os 
import fitz
import shutil

input_folder = r"/root/DSVA/data"
output_folder = r"/root/DSVA/processed_data"

def pdf_to_img(pdf_path,output_folder):
    doc=fitz.open(pdf_path)
    dpi=300

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
        output_image_name = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{page_num + 1}.png"
        output_image_path = os.path.join(output_folder, output_image_name)
        pix.save(output_image_path)
        print(f"saved page {page_num + 1} as {output_image_path}")

    doc.close()

def convert_pdf_to_img(root_folder, output_folder):
    for dirpath, dirname, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.pdf') or filename.endswith('.PDF'):
                pdf_path = os.path.join(dirpath, filename)

                relative_path = os.path.relpath(dirpath, root_folder)

                output_dir = os.path.join(output_folder, relative_path, os.path.splitext(filename)[0])
                os.makedirs(output_dir, exist_ok=True)

                pdf_to_img(pdf_path, output_dir)
                shutil.copy(pdf_path, output_dir)

convert_pdf_to_img(input_folder,output_folder)