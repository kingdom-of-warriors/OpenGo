import os
import re

def convert_txt_to_sgf_files(txt_file_path, output_dir="sgf_output"):
    """
    将txt文件中的每一行转换为单独的SGF文件
    """
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    sgf_count = 0
    with open(txt_file_path, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()
            if not line: continue
            if not (line.startswith('(') and line.endswith(')')): continue # 验证是否为的SGF
            filename = generate_filename(line, sgf_count + 1)
            output_path = os.path.join(output_dir, filename)
            with open(output_path, 'w', encoding='utf-8') as sgf_file: sgf_file.write(line) # 写入sgf
            sgf_count += 1
    
    print(f"\n转换完成! 共生成 {sgf_count} 个SGF文件在目录: {output_dir}")

def generate_filename(sgf_content, index):    
    black_player = extract_sgf_property(sgf_content, 'PB') # 提取黑棋玩家名
    white_player = extract_sgf_property(sgf_content, 'PW') # 提取白棋玩家名
    game_date = extract_sgf_property(sgf_content, 'DT') # 提取游戏日期
    filename_parts = [f"game_{index:03d}"]
    
    if black_player and white_player:
        black_clean = clean_filename(black_player)
        white_clean = clean_filename(white_player)
        filename_parts.append(f"{black_clean}_vs_{white_clean}")
    
    if game_date:
        date_clean = clean_filename(game_date)
        filename_parts.append(date_clean)
    
    filename = "_".join(filename_parts) + ".sgf"
    if len(filename) > 200:
        filename = f"game_{index:03d}.sgf"
    
    return filename

def extract_sgf_property(sgf_content, property_name):
    """
    从SGF内容中提取指定属性的值
    
    Args:
        sgf_content: SGF文件内容
        property_name: 要提取的属性名(如 PB, PW, DT, RE等)
    """
    pattern = rf'{property_name}\[([^\]]*)\]'
    match = re.search(pattern, sgf_content)
    return match.group(1) if match else None

def clean_filename(text):
    if not text: return ""
    illegal_chars = r'[<>:"/\\|?*]'
    cleaned = re.sub(illegal_chars, '_', text)
    cleaned = cleaned.strip()[:50]
    return cleaned


if __name__ == "__main__":
    txt_file_path = input("请输入txt文件路径: ").strip()
    output_directory = input("输出目录 (默认: sgf_output): ").strip()
    convert_txt_to_sgf_files(txt_file_path, output_directory)