
from great_textpectations import Textpectations
import textpectations_parsers as tp
import pprint as pp

def main():
    tt = Textpectations()
    USA = tt.load_text('pdfs/usa_1787.pdf', 'USA (1797)', parser=tp.pdf_parser)
    France = tt.load_text('pdfs/france_1791.pdf', 'France (1791)', parser=tp.pdf_parser)
    Mexico = tt.load_text('pdfs/mexico_1917.pdf', 'Mexico (1917)', parser=tp.pdf_parser)
    Germany1 = tt.load_text('pdfs/germany_1919.pdf', 'Germany 1919 (Weimar)', parser=tp.pdf_parser)
    Germany2 = tt.load_text('pdfs/germany_1949.pdf', 'Germany 1949 (Basic Law)', parser=tp.pdf_parser)
    Japan = tt.load_text('pdfs/japan_1947.pdf', 'Japan 1947 (post-WWII)', parser=tp.pdf_parser)
    Russia1 = tt.load_text('pdfs/russia_1918.pdf', 'Russia 1918 (Soviet)', parser=tp.pdf_parser)
    Russia2 =tt.load_text('pdfs/russia_1993.pdf', 'Russia 1993 (post-USSR)', parser=tp.pdf_parser)

    tt.similarity_scatterplot()

    pp.pprint(tt.data)




main()

