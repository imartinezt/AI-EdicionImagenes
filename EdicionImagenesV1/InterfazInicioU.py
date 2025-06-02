import asyncio
import os
import threading
import time
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, UnidentifiedImageError, ImageSequence

# Importamos la lógica de los nuevos modelos

import model1_update, model2_update

class ImageProcessingApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.lbl_image = None
        self.image_paths = None
        self.folder_path = None
        self.model_description = None
        self.model = None

        # --- Inicializar atributos para iconos ---
        self.icon_model1 = None
        self.icon_model2 = None
        self.icon_folder = None
        self.icon_exit = None
        self.icon_back = None

        self.title("Edición de Imágenes 4.0.0")
        self.geometry("900x600")
        self.resizable(False, False) # Deshabilita el redimensionamiento para mantener el layout fijo

        # Configuración de apariencia de CustomTkinter
        ctk.set_appearance_mode("System")  # "System" (default), "Dark", "Light"
        ctk.set_default_color_theme("blue")  # "blue" (default), "green", "dark-blue"

        # --- Frames principales: Sidebar y Contenido ---
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Frame para el menú lateral (sidebar) - Creado UNA SOLA VEZ
        self.frame_sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.frame_sidebar.grid(row=0, column=0, sticky="nswe")
        self.frame_sidebar.grid_rowconfigure(4, weight=1) # Permite que el botón de salir esté abajo

        # Frame principal para el contenido dinámico - Creado UNA SOLA VEZ
        self.main_frame = ctk.CTkFrame(self, corner_radius=0)
        self.main_frame.grid(row=0, column=1, sticky="nswe", padx=10, pady=10)
        self.main_frame.grid_columnconfigure(0, weight=1) # Centra el contenido en el main_frame

        # --- Cargar imágenes para iconos ---
        self.load_icons()

        # --- Inicializar el Sidebar UNA SOLA VEZ ---
        self.setup_sidebar()

        # Iniciar la pantalla de bienvenida en el main_frame
        self.show_welcome_screen()

    def load_icons(self):
        try:
            self.icon_model1 = ctk.CTkImage(Image.open("icons/recorte.png").resize((24, 24)))
            self.icon_model2 = ctk.CTkImage(Image.open("icons/fondo_blanco.png").resize((24, 24)))
            self.icon_folder = ctk.CTkImage(Image.open("icons/carpeta.png").resize((24, 24)))
            self.icon_exit = ctk.CTkImage(Image.open("icons/salir.png").resize((24, 24)))
            self.icon_back = ctk.CTkImage(Image.open("icons/atras.png").resize((24, 24)))
        except FileNotFoundError:
            messagebox.showwarning("Advertencia", "No se encontraron algunos iconos. Asegúrate de que están en la carpeta 'icons'.")
            self.icon_model1 = None
            self.icon_model2 = None
            self.icon_folder = None
            self.icon_exit = None
            self.icon_back = None

    def setup_sidebar(self):
        try:
            logo_image = Image.open("Liverpool_logo.svg.png")
            logo_image = logo_image.resize((180, 40))
            logo_photo = ctk.CTkImage(light_image=logo_image, dark_image=logo_image, size=(180, 40))

            lbl_logo = ctk.CTkLabel(self.frame_sidebar, image=logo_photo, text="")
            lbl_logo.pack(pady=(20, 20))
        except FileNotFoundError:
            lbl_logo = ctk.CTkLabel(self.frame_sidebar, text="Logo Faltante", font=ctk.CTkFont(size=14, weight="bold"))
            lbl_logo.pack(pady=(20, 20))

        ctk.CTkLabel(self.frame_sidebar, text="Selecciona un Modelo:", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10, 5))

        # Los botones del sidebar ahora llaman a select_model_and_show_details
        btn_model1 = ctk.CTkButton(self.frame_sidebar, text="  Modelo 1", image=self.icon_model1,
                                   compound="left", anchor="w",
                                   command=lambda: self.select_model_and_show_details("Modelo 1"), width=170, height=35)
        btn_model1.pack(pady=5, padx=10)

        btn_model2 = ctk.CTkButton(self.frame_sidebar, text="  Modelo 2", image=self.icon_model2,
                                   compound="left", anchor="w",
                                   command=lambda: self.select_model_and_show_details("Modelo 2"), width=170, height=35)
        btn_model2.pack(pady=5, padx=10)

        btn_exit = ctk.CTkButton(self.frame_sidebar, text="  Salir", image=self.icon_exit,
                                 compound="left", anchor="w",
                                 command=self.quit, width=170, height=35)
        btn_exit.pack(side="bottom", pady=(10, 20), padx=10)


    def clear_main_frame(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

    def show_welcome_screen(self):
        self.clear_main_frame()

        lbl_title = ctk.CTkLabel(self.main_frame, text="Bienvenido a Edición de Imágenes 4.0.0",
                                 font=ctk.CTkFont(family="Montserrat", size=24, weight="bold"))
        lbl_title.pack(pady=50)

        welcome_text = (
            "Para empezar, selecciona uno de los modelos de procesamiento de imagen del menú lateral.\n\n"
            "Cada modelo ofrece funcionalidades únicas para optimizar tus imágenes.\n\n"
            "¡Esperamos que disfrutes la experiencia!"
        )
        lbl_welcome_info = ctk.CTkLabel(self.main_frame, text=welcome_text,
                                        font=ctk.CTkFont(family="Open Sans", size=14),
                                        wraplength=600, justify="center") # Alineación a la izquierda para las descripciones
        lbl_welcome_info.pack(pady=20, padx=20) # Añadir padx para evitar que el texto toque los bordes

        footer = ctk.CTkLabel(self.main_frame, text="© 2024 Equipo de AI", font=ctk.CTkFont(family="Open Sans", size=10))
        footer.pack(side="bottom", pady=15)

    def select_model_and_show_details(self, model_name):
        self.model = model_name
        if model_name == "Modelo 1":
            self.model_description = (
                "**Recorte y Puesta a Punto:**\n"
                "- Recorte centrado ✂️\n"
                "- Lienzo a **940 px de ancho x 1215 px de alto** 📐\n"
                "- Resolución **72 dpis** 📸"
            )
        elif model_name == "Modelo 2":
            self.model_description = (
                "**Quitar Fondo, Recorte y Puesta a Punto:**\n"
                "- Recorte centrado ✂️\n"
                "- Lienzo a **940px de ancho x 1215px de alto** 📐\n"
                "- Resolución **72 dpis** 💻\n"
                "- Remueve el fondo 🌫️\n"
                "- Rellenar con fondo blanco ⚪️"
            )
        self.clear_main_frame()
        self.show_selected_model_details()

    def show_selected_model_details(self):
        # Esta función asume que self.model y self.model_description ya están establecidos.

        lbl_model_title = ctk.CTkLabel(self.main_frame, text=self.model,
                                       font=ctk.CTkFont(family="Montserrat", size=24, weight="bold"))
        lbl_model_title.pack(pady=(20, 10))

        # *** CAMBIO AQUÍ: Usamos CTkLabel en lugar de CTkTextbox ***
        lbl_description = ctk.CTkLabel(self.main_frame, text=self.model_description,
                                       font=ctk.CTkFont(family="Open Sans", size=16),
                                       wraplength=550, # Asegura que el texto se envuelva
                                       justify="left") # Alinea el texto a la izquierda
        lbl_description.pack(pady=20, padx=20) # Añade padding para separación

        btn_select_folder = ctk.CTkButton(self.main_frame, text="  Seleccionar Carpeta", image=self.icon_folder,
                                          compound="left", command=self.load_images, width=280, height=45,
                                          font=ctk.CTkFont(size=16, weight="bold"))
        btn_select_folder.pack(pady=(30, 10))

        btn_back = ctk.CTkButton(self.main_frame, text="  Atrás", image=self.icon_back,
                                 compound="left", command=self.show_welcome_screen, width=280, height=45,
                                 font=ctk.CTkFont(size=16))
        btn_back.pack(pady=5)

        footer = ctk.CTkLabel(self.main_frame, text="© 2024 Equipo de AI", font=ctk.CTkFont(family="Open Sans", size=10))
        footer.pack(side="bottom", pady=15)

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
        self.clear_main_frame()

        lbl_loading = ctk.CTkLabel(self.main_frame, text="¡Casi listo! Procesando tus imágenes...",
                                    font=ctk.CTkFont(family="Montserrat", size=18, weight="bold"))
        lbl_loading.pack(pady=40)

        try:
            loading_image = Image.open("spinner.gif")
            frames = []

            for frame in ImageSequence.Iterator(loading_image):
                frame = frame.convert("RGBA")
                frame = frame.resize((300, 300))
                frames.append(ctk.CTkImage(light_image=frame, dark_image=frame, size=(300, 300)))

            lbl_image_gif = ctk.CTkLabel(self.main_frame, text="")
            lbl_image_gif.pack(pady=20)
            self.lbl_image = lbl_image_gif

            def update_frame(index, lbl=lbl_image_gif):
                if lbl.winfo_exists():
                    fotograma = frames[index]
                    lbl.configure(image=fotograma)
                    self.after(100, update_frame, (index + 1) % len(frames), lbl)
                else:
                    print("Etiqueta de imagen de carga destruida.")
            self.after(0, update_frame, 0, lbl_image_gif)

        except (FileNotFoundError, UnidentifiedImageError):
            lbl_loading.configure(text="Procesando imágenes, por favor espere...\n(No se encontró animación de carga)")
            progressbar = ctk.CTkProgressBar(self.main_frame, mode='indeterminate', width=200)
            progressbar.pack(pady=20)
            progressbar.start()

        footer = ctk.CTkLabel(self.main_frame, text="© 2024 Equipo de AI", font=ctk.CTkFont(family="Open Sans", size=10))
        footer.pack(side="bottom", pady=15)

    def process_images(self):
        try:
            start_time = time.time()
            output_folder = os.path.expanduser(
                "~/Desktop/SalidaModel1AI" if self.model == "Modelo 1" else "~/Desktop/SalidaModel2AI")

            os.makedirs(output_folder, exist_ok=True)

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
        await model1_update.process_images_in_folder(self.folder_path, output_folder)

    def process_model2_images(self, output_folder):
        model2_update.process_images_in_folder(self.folder_path, output_folder)

    def show_completion_message(self, elapsed_time):
        if self.lbl_image and self.lbl_image.winfo_exists():
            self.lbl_image.destroy()

        messagebox.showinfo("Proceso Completado", f"¡Edición finalizada con éxito!\n"
                                                  f"Tiempo transcurrido: {elapsed_time:.2f} segundos.")
        self.ask_load_more_images()

    def ask_load_more_images(self):
        result = messagebox.askquestion("Cargar más imágenes", "¿Deseas procesar más imágenes?")
        if result == "yes":
             self.load_images()# Si, ir a seleccionar carpeta
        else:
            self.show_welcome_screen() # No, volver a la pantalla de bienvenida (que no duplica el sidebar)

if __name__ == "__main__":
    app = ImageProcessingApp()
    app.mainloop()