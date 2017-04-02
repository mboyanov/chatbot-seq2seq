from flask import Flask
from chatbot import decode, FLAGS
import tensorflow as tf
from flask import request
from flask.ext.cors import CORS, cross_origin
import os

import json
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/")
@cross_origin()
def bot():
    msg = request.args.get('message', '')
    response = {
        "message": responder(msg)
    }
    if 'uuid' in request.args:
        chatpath = os.path.join(FLAGS.train_dir, "%s.chat" % request.args.get('uuid'))
        with open(chatpath, 'a') as out:
            out.write("User: %s\nQLBot: %s\n" % (msg, response['message']))
    return json.dumps(response)


def responder(s):
    return "hi"

if __name__ == "__main__":
    with tf.Session() as sess:
        responder = decode(sess)
        app.run(host="0.0.0.0")

