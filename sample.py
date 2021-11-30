import text_mining

target_dir = './text_report'

text_mining.show_conetwork(text_dir=target_dir,
                           edge_threshold=5,
                           k=0.6,
                           sw_list=['当時', '今', '人', '着', '流行っ', 'こと', 'とき', 'もの', 'ところ', 'いう', 'お', 'しれ', '方', '〓'])

# text_mining.show_word_rank(text_dir=target_dir,
#                            n_top=50,
#                            sw_list=['こと', 'とき', 'もの', 'ところ', 'いう', 'お', 'しれ', '方', '〓'],
#                            font_size=6,
#                            bottom_position=0.20)
#
# text_mining.show_wordcloud(text_dir=target_dir,
#                            sw_list=['当時', '今', '人', '着る', '流行る', 'こと', 'とき', 'もの', 'ところ', 'いう', 'お', 'しれ', '方', '〓'],
#                            font_path="./fonts/ipaexg.ttf")




