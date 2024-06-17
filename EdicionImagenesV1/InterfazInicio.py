import tkinter as tk
import os
import threading
import asyncio
import time
import model1
import model2

from tkinter import ttk, filedialog, messagebox
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageTk, UnidentifiedImageError, ImageSequence

"""
@Autor: Iván Martínez Trejo, 
| Foro Fotografico | Front end | Integracion de Modelos. 
--> Front end para la integracion de los modelos Recorte puesta punto y sustitucion background. 
Update 2.0

"""


class ImageProcessingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.lbl_image = None
        self.image_paths = None
        self.folder_path = None
        self.model_description = None
        self.model = None
        self.title("Edición de Imágenes 2.5")
        self.geometry("400x500")
        self.config(bg="#FFC0CB")  # Fondo rosa claro
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.style.configure("TButton", background="#FF69B4", foreground="white", font=("Helvetica", 12, "bold"))
        self.style.map("TButton",
                       background=[('active', '#FF1493')],
                       foreground=[('active', 'white')])
        self.model_selection()

    def model_selection(self):
        self.clear_window()

        # logotipo
        logo_image = Image.open("Liverpool_logo.svg.png")
        logo_image = logo_image.resize((200, 50))
        logo_photo = ImageTk.PhotoImage(logo_image)
        lbl_logo = tk.Label(self, image=logo_photo, bg="#FFC0CB")
        lbl_logo.image = logo_photo
        lbl_logo.pack(pady=10)

        lbl = ttk.Label(self, text="Edición de Imágenes 2.5", font=("Helvetica", 16, "bold"), background="#FFC0CB")
        lbl.pack(pady=10)

        frame_buttons = ttk.Frame(self, style="TFrame", relief="flat")
        frame_buttons.pack(pady=10)

        btn_model1 = ttk.Button(frame_buttons, text="Modelo 1", command=self.select_model1, style="TButton")
        btn_model1.pack(side=tk.LEFT, padx=20)

        btn_model2 = ttk.Button(frame_buttons, text="Modelo 2", command=self.select_model2, style="TButton")
        btn_model2.pack(side=tk.RIGHT, padx=20)

        btn_exit = ttk.Button(self, text="Salir", command=self.quit, style="TButton")
        btn_exit.pack(pady=20)

        #  pie de página
        footer = ttk.Label(self, text="© 2024 Equipo de AI", background="#FFC0CB", font=("Helvetica", 10))
        footer.pack(side=tk.BOTTOM, pady=10)

    def clear_window(self):
        for widget in self.winfo_children():
            widget.destroy()

    def select_model1(self):
        self.model = "Modelo 1"
        self.model_description = (
            "Recorte y puesta a punto:\n"
            "- Recorte centrado\n"
            "- Lienzo a 940 px de ancho x 1215 px de alto\n"
            "- Resolución 72 dpis"
        )
        self.select_folder()

    def select_model2(self):
        self.model = "Modelo 2"
        self.model_description = (
            "Quitar fondo, recorte y puesta a punto:\n"
            "- Recorte centrado\n"
            "- Lienzo a 940 px de ancho x 1215 px de alto\n"
            "- Resolución 72 dpis\n"
            "- Remueve el fondo\n"
            "- Rellenar con fondo blanco (#FFFFFF)"
        )
        self.select_folder()

    def select_folder(self):
        self.clear_window()

        lbl_model = ttk.Label(self, text=self.model, font=("Helvetica", 14, "bold"), background="#FFC0CB")
        lbl_model.pack(pady=10)

        lbl_description = ttk.Label(self, text=self.model_description, background="#FFC0CB")
        lbl_description.pack(pady=10)

        btn_select_folder = ttk.Button(self, text="Seleccionar Carpeta", command=self.load_images, style="TButton")
        btn_select_folder.pack(pady=10)

        btn_back = ttk.Button(self, text="Atrás", command=self.model_selection, style="TButton")
        btn_back.pack(pady=5)

    def load_images(self):
        self.folder_path = filedialog.askdirectory()
        if not self.folder_path:
            messagebox.showwarning("Advertencia", "Debes seleccionar una carpeta.")
            return

        self.image_paths = [os.path.join(self.folder_path, file) for file in os.listdir(self.folder_path) if
                            file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        if not self.image_paths:
            messagebox.showwarning("Advertencia", "No se encontraron imágenes en la carpeta seleccionada.")
            return

        self.show_loading_screen()
        threading.Thread(target=self.process_images).start()

    def show_loading_screen(self):
        self.clear_window()

        lbl_loading = ttk.Label(self, text="Procesando imágenes, por favor espere...", background="#FFC0CB")
        lbl_loading.pack(pady=10)

        # animación de carga si existe
        try:
            loading_image = Image.open("loading.gif")
            frames = [ImageTk.PhotoImage(frame) for frame in ImageSequence.Iterator(loading_image)]
            lbl_image = ttk.Label(self, background="#FFC0CB")
            lbl_image.pack(pady=10)
            self.lbl_image = lbl_image

            def update_frame(index, lbl=lbl_image):
                if lbl.winfo_exists():
                    frame = frames[index]
                    lbl.configure(image=frame)
                    lbl.image = frame
                    self.after(100, update_frame, (index + 1) % len(frames), lbl)
                else:
                    print("Label no existe más, deteniendo la actualización de frames.")

            self.after(0, update_frame, 0, lbl_image)
        except (FileNotFoundError, UnidentifiedImageError):
            lbl_loading.config(text="Procesando imágenes, por favor espere...")

    def process_images(self):
        try:
            start_time = time.time()

            if self.model == "Modelo 1":
                thread_executor = ThreadPoolExecutor(max_workers=10)
                process_executor = ThreadPoolExecutor(max_workers=14)
                asyncio.run(model1.process_images_async(self.image_paths, thread_executor=thread_executor,
                                                        process_executor=process_executor))
                thread_executor.shutdown(wait=True)
                process_executor.shutdown(wait=True)
            elif self.model == "Modelo 2":
                asyncio.run(model2.process_images(self.folder_path))

            end_time = time.time()
            elapsed_time = end_time - start_time
            self.show_completion_message(elapsed_time)
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error durante el procesamiento de imágenes: {str(e)}")
            print(f"Error procesando las imágenes: {str(e)}")

    def show_completion_message(self, elapsed_time):
        messagebox.showinfo("Proceso Completado", f"El proceso de procesamiento de imágenes ha sido completado.\n"
                                                  f"Tiempo transcurrido: {elapsed_time:.2f} segundos.")
        self.ask_load_more_images()

    def ask_load_more_images(self):
        result = messagebox.askquestion("Cargar más imágenes", "¿Desea cargar más imágenes?")
        if result == "yes":
            self.select_folder()
        else:
            self.model_selection()


if __name__ == "__main__":
    app = ImageProcessingApp()
    app.mainloop()
