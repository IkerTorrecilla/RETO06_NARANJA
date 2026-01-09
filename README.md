# RETO 06 – Equipo Naranja
# Descripción del proyecto

Este repositorio contiene el desarrollo del **Reto 06 del Equipo Naranja**, cuyo objetivo es realizar un proceso completo de **análisis de datos**, incluyendo:

- Limpieza y preprocesamiento de datos
- Análisis exploratorio
- Modelado y técnicas de clustering
- Evaluación y visualización de resultados
- Scrapping y tecnicas de NLP 

El proyecto se ha desarrollado en Python, en la version 3.11 utilizando el entorno de Conda.

## Estructura del proyecto

```text
RET006_NARANJA/
│
├── Datos/
│ ├── DatosOriginales/ # Datos en bruto (CSV originales) -> Esta añadido en el gitignore para que no se suba
│ ├── DatosTransformados/ # Datos tras el proceso de limpieza y transformación
│ ├── Resultados/-> Sirve para el modelado y para la comparacion de los algoritmos del clustering ya que contiene la tabla de la comparacion de las metricas de los algoritmos de clustering y los dataset con los datos meteorologicos y los cluster asignados  

├── Scripts/
│ ├── Script_Analisis_Exploratorio/ # Análisis exploratorio inicial de los datos
│ │
│ ├── Scripts_BigDatayEcosistemaDigital/ # Scripts relacionados con Big Data y ecosistema digital
│ │
│ ├── Scripts_Limpieza_Datos/ # Funciones y scripts para la limpieza de datos
│ │
│ ├── Scripts_Modelado_Clustering/ # Notebooks de modelado y tecnicas de clustering
│ │
│ ├── Scripts_Scrapping_NLP/ # Scripts de scraping y procesamiento de lenguaje natural
│
├── environment.yml # Definición del entorno Conda del proyecto
├── .gitignore # Archivos y carpetas ignoradas por Git
├── README.md # Documentación principal del proyecto

```

## Datos Transformados

Tras consultarlo con los Project Managers (PMs), se ha confirmado que no existe inconveniente en incluir los datos transformados dentro del repositorio.

Por este motivo, se ha decidido mantener los datos transformados con el objetivo de facilitar la revisión y evaluación del proyecto, evitando que el profesorado tenga que ejecutar nuevamente todo el proceso de limpieza, modelado y clustering.

Cabe destacar que la ejecución completa del pipeline de modelado y clustering puede tardar aproximadamente 2 horas, por lo que disponer de los datos ya procesados agiliza considerablemente la revisión del trabajo.

## Para ejecutar el proyecto es necesario:

- Conda 
- Python **3.11**
- Dependencias especificadas en **environment.yml**

## Creacion del entorno
conda env create -f environment.yml
conda activate equipo_naranja_reto6