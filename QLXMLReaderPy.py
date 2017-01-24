import xml.etree.ElementTree as ET

def read(data_path):
    tree = ET.parse(data_path)
    root = tree.getroot()
    for thread in root:
        question = thread.find('RelQuestion').find('RelQClean').text
        answers = []
        for c in thread.findall('RelComment'):
            answer = c.find('RelCClean').text
            answers.append((answer, c.attrib['RELC_RELEVANCE2RELQ'] == 'Good'))
        yield((question, answers))