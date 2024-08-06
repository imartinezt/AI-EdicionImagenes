import asyncio
import os
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, UnidentifiedImageError, ImageSequence
import model1
import model2


class ImageProcessingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.lbl_image = None
        self.image_paths = None
        self.folder_path = None
        self.model_description = None
        self.model = None
        self.title("Edición de Imágenes 3.0.1")
        self.geometry("440x607")
        self.configure(bg="#F3E5F5")
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.style.configure("TButton", background="#CE93D8", foreground="white", font=("Helvetica", 14, "bold"))
        self.style.map("TButton",
                       background=[('active', '#BA68C8')],
                       foreground=[('active', 'white')])
        self.model_selection()

    def model_selection(self):
        self.clear_window()

        # logotipo
        logo_image = Image.open("Liverpool_logo.svg.png")
        logo_image = logo_image.resize((320, 70), Image.LANCZOS)
        logo_photo = ImageTk.PhotoImage(logo_image)
        lbl_logo = tk.Label(self, image=logo_photo, bg="#F3E5F5")
        lbl_logo.image = logo_photo
        lbl_logo.pack(pady=20)

        lbl_title = ttk.Label(self, text="Edición de Imágenes 3.0", font=("Helvetica", 24, "bold"),
                              background="#F3E5F5")
        lbl_title.pack(pady=10)

        frame_buttons = ttk.Frame(self, style="TFrame", relief="flat")
        frame_buttons.pack(pady=20)

        btn_model1 = ttk.Button(frame_buttons, text="Modelo 1", command=self.select_model1, style="TButton", width=20)
        btn_model1.pack(side=tk.LEFT, padx=20)

        btn_model2 = ttk.Button(frame_buttons, text="Modelo 2", command=self.select_model2, style="TButton", width=20)
        btn_model2.pack(side=tk.RIGHT, padx=20)

        btn_exit = ttk.Button(self, text="Salir", command=self.quit, style="TButton")
        btn_exit.pack(pady=20)

        # pie de página
        footer = ttk.Label(self, text="© 2024 Equipo de AI", background="#F3E5F5", font=("Helvetica", 12))
        footer.pack(side=tk.BOTTOM, pady=10)

    def clear_window(self):
        for widget in self.winfo_children():
            widget.destroy()

    def select_model1(self):
        self.model = "Modelo 1"
        self.model_description = (
            "\t\b Recorte y puesta a punto:\n"
            "\t\b- Recorte centrado ✂️\n"
            "\t\b- Lienzo a 940 px de ancho x 1215 px de alto 📐\n"
            "\t\b- Resolución 72 dpis 📸"
        )
        self.select_folder()

    def select_model2(self):
        self.model = "Modelo 2"
        self.model_description = (
            "\t\bQuitar fondo, recorte y puesta a punto:\n"
            "\t\b- Recorte centrado ✂️\n"
            "\t\b- Lienzo a 940px de ancho x 1215px de alto 📐\n"
            "\t\b- Resolución 72 dpis 💻\n"
            "\t\b- Remueve el fondo 🌫️\n"
            "\t\b- Rellenar con fondo blanco ⚪️"
        )
        self.select_folder()

    def select_folder(self):
        self.clear_window()

        lbl_model = ttk.Label(self, text=self.model, font=("Helvetica", 20, "bold"), background="#F3E5F5")
        lbl_model.pack(pady=10)

        frame_description = ttk.Frame(self, style="TFrame", relief="flat", padding=10)
        frame_description.pack(pady=10, padx=10, fill='x')

        lbl_description = ttk.Label(frame_description, text=self.model_description, background="#E1BEE7",
                                    font=("Helvetica", 14), padding=10, wraplength=500)
        lbl_description.pack()

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

        lbl_loading = ttk.Label(self, text="Procesando imágenes, por favor espere...", background="#F3E5F5",
                                font=("Helvetica", 16))
        lbl_loading.pack(pady=10)

        # animación de carga si existe
        try:
            loading_image = Image.open("loading.gif")
            frames = []

            for frame in ImageSequence.Iterator(loading_image):
                frame = frame.convert("RGBA")
                frame = frame.resize((940, 660), Image.LANCZOS)
                frames.append(ImageTk.PhotoImage(frame))

            lbl_image = ttk.Label(self, background="#F3E5F5")
            lbl_image.pack(pady=10)
            self.lbl_image = lbl_image

            def update_frame(index, lbl=lbl_image):
                if lbl.winfo_exists():
                    fotograma = frames[index]
                    lbl.configure(image=fotograma)
                    lbl.image = fotograma
                    self.after(100, update_frame, (index + 1) % len(frames), lbl)
                else:
                    print(":D")

            self.after(0, update_frame, 0, lbl_image)
        except (FileNotFoundError, UnidentifiedImageError):
            lbl_loading.config(text="Procesando imágenes, por favor espere...")

    def process_images(self):
        try:
            start_time = time.time()
            output_folder = os.path.expanduser(
                "~/Desktop/SalidaModel1AI" if self.model == "Modelo 1" else "~/Desktop/SalidaModel2AI")

            if self.model == "Modelo 1":
                asyncio.run(self.process_model1_images(output_folder))
            elif self.model == "Modelo 2":
                self.process_model2_images(output_folder)

            end_time = time.time()
            elapsed_time = end_time - start_time
            self.show_completion_message(elapsed_time)
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error durante el procesamiento de imágenes: {str(e)}")
            print(f"Error procesando las imágenes: {str(e)}")

    async def process_model1_images(self, output_folder):
        await model1.process_images_in_folder(self.folder_path, output_folder)

    def process_model2_images(self, output_folder):
        model2.process_images_in_folder(self.folder_path, output_folder)

    def show_completion_message(self, elapsed_time):
        messagebox.showinfo("Proceso Completado", f"El proceso de Edición de imágenes ha sido completado.\n"
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
