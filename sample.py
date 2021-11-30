import text_mining

target_dir = './text_report'

text_mining.show_conetwork(text_dir=target_dir,
                           word="ファッション",
                           edge_threshold=2,
                           k=0.1,
                           sw_list=['こと', 'とき', 'もの', 'ところ', 'いう', 'お', 'しれ'])

# text_mining.show_word_rank(text_dir=target_dir,
#                            n_top=50,
#                            sw_list=['こと', 'とき', 'もの', 'ところ', 'いう', 'お', 'しれ'],
#                            font_size=6,
#                            bottom_position=0.20)

# text_mining.show_wordcloud(text_dir=target_dir,
#                            sw_list=['こと', 'とき', 'もの', 'ところ', 'いう', 'お', 'しれ'],
#                            font_path="./fonts/ipaexg.ttf")




