
from great_textpectations import Textpectations
import textpectations_parsers as tp
import pprint as pp

def main():
    tt = Textpectations()
    tt.load_text('test.txt', 'A')
    tt.load_text('pdfs/france_1791.pdf', 'B', parser=tp.pdf_parser)

    pp.pprint(tt.data)




main()

