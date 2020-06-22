import os, random, sys
from xml.dom.minidom import Document, parse
from PIL import Image

CLASSES = ['stop_sign', 'Leopards', 'Motorbikes', 'starfish', 'lamp', 'airplanes', 'brain', 
    'accordion', 'Faces', 'umbrella', 'helicopter', 'ewer', 'Faces_easy', 'camera', 'flamingo_head', 
    'crab', 'barrel', 'scorpion', 'sea_horse', 'okapi', 'cup', 'sunflower', 'dolphin', 'yin_yang', 
    'dollar_bill', 'wrench', 'windsor_chair', 'inline_skate', 'chair', 'wild_cat', 'chandelier', 
    'trilobite', 'watch', 'schooner', 'euphonium', 'llama', 'brontosaurus', 'kangaroo', 'saxophone', 
    'ketch', 'butterfly', 'rhino', 'hawksbill', 'pyramid', 'crocodile', 'revolver', 'octopus', 'car_side', 
    'electric_guitar', 'buddha', 'dalmatian', 'grand_piano', 'garfield', 'lobster', 'cougar_face', 'binocular', 
    'ibis', 'wheelchair', 'joshua_tree', 'bonsai', 'ferry', 'anchor', 'lotus', 'pizza', 'mandolin', 'pagoda', 
    'BACKGROUND_Google', 'gramophone', 'laptop', 'scissors', 'soccer_ball', 'nautilus', 'minaret', 'crocodile_head', 
    'strawberry', 'emu', 'mayfly', 'gerenuk', 'elephant', 'bass', 'water_lilly', 'snoopy', 'ant', 'platypus', 
    'menorah', 'stegosaurus', 'crayfish', 'cannon', 'beaver', 'tick', 'headphone', 'rooster', 'cellphone', 'flamingo', 
    'panda', 'dragonfly', 'pigeon', 'hedgehog', 'cougar_body', 'metronome', 'stapler', 'ceiling_fan']

def writeXml(filename, cls):
    doc = Document()
    root = doc.createElement('image')
    doc.appendChild(root)
    node = doc.createElement('class')
    node_value = doc.createTextNode(cls)
    node.appendChild(node_value)
    root.appendChild(node)

    path = os.path.join("./dataset/annotations/", filename)
    with open(path, 'w') as f:
        f.write(doc.toprettyxml(indent='\t'))

def readXml(filepath):
    domTree = parse(filepath)
    rootNode = domTree.documentElement
    object_node = rootNode.getElementsByTagName("class")[0]
    object_cls = object_node.childNodes[0].data
    return object_cls

def resize(img_path):
    img = Image.open(img_path)
    out = img.resize((224, 224), Image.ANTIALIAS)
    out.save(img_path)

def get_class():
    classes = {}
    anno_list = os.listdir("./dataset/annotations")
    count = 0
    for anno in anno_list:
        xml_path = os.path.join('./dataset/annotations', anno)
        object_cls = readXml(xml_path)
        print(object_cls)
        if object_cls not in classes.keys():
            classes[object_cls] = 0
        classes[object_cls] += 1
        count += 1

    print(classes.keys())
        
if __name__ == "__main__":
    get_class()
