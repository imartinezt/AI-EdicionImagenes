# Edicion de imagenes

Este repositorio contiene los scripts para el levantamiento de un front que realiza ediciones de imagenes utilizando alguno de dos modelos


# Instrucciones de instalacion

## Local
Para ejecutar este ambiente de manera local se necesitara correr el siguiente comando

python InterfazInicio.py

## Creacion de ejecutable

### Windows

Para esto, se necesitara previamente tener instalada la libreria pyinstaller. A partir de ahi se ejecutara el siguiente comando

pyinstaller --onefile --add-data "Liverpool_logo.svg.png;img" --add-data "loading.gif;img" InterfazInicio.py -D

Enseguida se vera una carpeta llamada dist dentro del directorio.
A continuacion hay que copiar el archivo Liverpool_logo.png dentro de la carpeta dist/InterfazInicio
Luego, habrá que buscar el siguiente archivo: "onnxruntime_providers_shared.dll" dentro de la distribución de python. Por ejemplo en la carpeta

```
C:\Users\GOCOTAM\AppData\Local\anaconda3\Lib\site-packages\onnxruntime\capi
```

Finalmente, habrá que copiar dicho documento en la carpeta
```
dist\InterfazInicio\_internal\onnxruntime\capi
```