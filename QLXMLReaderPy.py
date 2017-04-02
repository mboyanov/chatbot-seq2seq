import xml.etree.ElementTree as ET


def read(data_path, question_text_property="RelQClean", comment_text_property="RelCClean"):
    tree = ET.parse(data_path)
    root = tree.getroot()
    for i, thread in enumerate(root):
        question = thread.find('.//RelQuestion').find(question_text_property).text
        answers = []
        for c in thread.findall('.//RelComment'):
            answer = c.find(comment_text_property).text
            answers.append((answer, c.attrib['RELC_RELEVANCE2RELQ'] == 'Good', c.attrib['RELC_ID']))
        yield((question, answers))