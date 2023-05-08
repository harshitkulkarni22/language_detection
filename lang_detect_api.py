from flask import Flask, request, jsonify
import fasttext

class LanguageIdentification:
    def __init__(self, model_path):
        self.model = fasttext.load_model(model_path)

    def predict_language(self, text):
        predictions = self.model.predict(text, k=2)[0] # top 2 matching languages
        return predictions

app = Flask(__name__)
language_identifier = LanguageIdentification('lid.176.ftz')

@app.route('/detect-language', methods=['POST'])
def detect_language():
    if request.is_json:
        content = request.get_json()
        text = content['text']
        lang_predictions = language_identifier.predict_language(text)
        if lang_predictions:
            if any('__label__id' in label for label in lang_predictions):
                language = 'Bahasa Indonesia'
            elif any('__label__en' in label for label in lang_predictions):
                language = 'English'
            else:
                language = 'Unknown language'
            return jsonify({'language': language})
        else:
            return jsonify({'error': 'Unable to detect language'})
    else:
        return jsonify({'error': 'Invalid request'})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')