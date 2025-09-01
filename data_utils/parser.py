import os
import re

def convert_txt_to_sgf_files(txt_file_path, output_dir="sgf_output"):
    """
    将txt文件中的每一行转换为单独的SGF文件
    
    Args:
        txt_file_path: 输入的txt文件路径
        output_dir: 输出SGF文件的目录
    """
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    sgf_count = 0
    
    try:
        with open(txt_file_path, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, 1):
                line = line.strip()
                
                # 跳过空行
                if not line:
                    continue
                
                # 验证是否为有效的SGF格式
                if not (line.startswith('(') and line.endswith(')')):
                    print(f"警告: 第 {line_number} 行不是有效的SGF格式，跳过")
                    continue
                
                # 尝试从SGF内容中提取游戏信息用于文件命名
                filename = generate_filename(line, sgf_count + 1)
                
                # 写入SGF文件
                output_path = os.path.join(output_dir, filename)
                with open(output_path, 'w', encoding='utf-8') as sgf_file:
                    sgf_file.write(line)
                
                sgf_count += 1
                print(f"已创建: {output_path}")
    
    except FileNotFoundError:
        print(f"错误: 找不到文件 {txt_file_path}")
        return
    except Exception as e:
        print(f"错误: {e}")
        return
    
    print(f"\n转换完成！共生成 {sgf_count} 个SGF文件在目录: {output_dir}")

def generate_filename(sgf_content, index):
    """
    根据SGF内容生成有意义的文件名
    
    Args:
        sgf_content: SGF文件内容
        index: 文件序号
    
    Returns:
        生成的文件名
    """
    
    black_player = extract_sgf_property(sgf_content, 'PB') # 提取黑棋玩家名
    white_player = extract_sgf_property(sgf_content, 'PW') # 提取白棋玩家名
    game_date = extract_sgf_property(sgf_content, 'DT') # 提取游戏日期
    result = extract_sgf_property(sgf_content, 'RE') # 提取比赛结果
    
    # 构建文件名
    filename_parts = [f"game_{index:03d}"]
    
    if black_player and white_player:
        # 清理玩家名中的特殊字符
        black_clean = clean_filename(black_player)
        white_clean = clean_filename(white_player)
        filename_parts.append(f"{black_clean}_vs_{white_clean}")
    
    if game_date:
        date_clean = clean_filename(game_date)
        filename_parts.append(date_clean)
    
    filename = "_".join(filename_parts) + ".sgf"
    
    # 确保文件名不会太长
    if len(filename) > 200:
        filename = f"game_{index:03d}.sgf"
    
    return filename

def extract_sgf_property(sgf_content, property_name):
    """
    从SGF内容中提取指定属性的值
    
    Args:
        sgf_content: SGF文件内容
        property_name: 要提取的属性名(如 PB, PW, DT, RE等)
    
    Returns:
        属性值，如果没找到返回None
    """
    pattern = rf'{property_name}\[([^\]]*)\]'
    match = re.search(pattern, sgf_content)
    return match.group(1) if match else None

def clean_filename(text):
    """
    清理文件名中的非法字符
    
    Args:
        text: 原始文本
    
    Returns:
        清理后的文本
    """
    if not text:
        return ""
    
    # 移除或替换文件名中的非法字符
    illegal_chars = r'[<>:"/\\|?*]'
    cleaned = re.sub(illegal_chars, '_', text)
    
    # 移除首尾空格并限制长度
    cleaned = cleaned.strip()[:50]
    
    return cleaned

def analyze_txt_file(txt_file_path):
    """
    分析txt文件，显示基本信息
    
    Args:
        txt_file_path: txt文件路径
    """
    try:
        with open(txt_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        total_lines = len(lines)
        non_empty_lines = len([line for line in lines if line.strip()])
        
        print(f"文件分析:")
        print(f"总行数: {total_lines}")
        print(f"非空行数: {non_empty_lines}")
        
        # 检查前几行的格式
        print(f"\n前3行预览:")
        for i, line in enumerate(lines[:3]):
            line = line.strip()
            if line:
                is_sgf = line.startswith('(') and line.endswith(')')
                print(f"第{i+1}行: {'✓' if is_sgf else '✗'} SGF格式 (长度: {len(line)})")
                if len(line) > 100:
                    print(f"  内容: {line[:100]}...")
                else:
                    print(f"  内容: {line}")
            else:
                print(f"第{i+1}行: 空行")
    
    except Exception as e:
        print(f"分析文件时出错: {e}")

if __name__ == "__main__":
    # 使用示例
    txt_file_path = input("请输入txt文件路径: ").strip()
    
    if not txt_file_path:
        # 如果没有输入，尝试在当前目录寻找txt文件
        import glob
        txt_files = glob.glob("*.txt")
        if txt_files:
            txt_file_path = txt_files[0]
            print(f"使用找到的文件: {txt_file_path}")
        else:
            print("没有找到txt文件")
            exit(1)
    
    if not os.path.exists(txt_file_path):
        print(f"文件不存在: {txt_file_path}")
        exit(1)
    
    # 分析文件
    analyze_txt_file(txt_file_path)
    
    # 询问是否继续转换
    response = input("\n是否继续转换? (y/n): ").strip().lower()
    if response in ['y', 'yes', '是']:
        output_directory = input("输出目录 (默认: sgf_output): ").strip() or "sgf_output"
        convert_txt_to_sgf_files(txt_file_path, output_directory)
    else:
        print("取消转换")