# Virtual Environment 
## create virtual environment 
`python3 -m venv venv`
## activate virtual environment
`source venv/bin/activate`
## install dependencies
`pip install -r requirements.txt`

# Process the Image(s)
## Single image
`python image2text.py path/to/image.png --lang eng`
### example
`python image2text.py example_images/example_image1.png --lang eng`


## Folder of images (writes one .txt per image)
`python image2text.py path/to/folder --lang eng`
### example 
`python image2text.py example_images --lang eng`

## German without preprocessing
`python image2text.py scans/ --lang deu --no-preprocess`


