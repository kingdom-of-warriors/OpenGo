import os
from pathlib import Path

def process_files_to_sgf(input_folder: str, output_folder: str):
    """
    递归地遍历输入文件夹，找到所有没有后缀名的文件。
    对于找到的每个文件，逐行读取，并将每一行都转换成一个独立的 .sgf 文件。
    """
    input_path = Path(input_folder).resolve(strict=True)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
        
    print(f"输入文件夹: {input_path}")
    print(f"输出文件夹: {output_path.resolve()}")
    
    total_sgf_count = 0
    processed_files_count = 0
    
    # 递归遍历所有文件和文件夹
    for file_path in input_path.rglob('*'):
        if file_path.is_file() and not file_path.suffix: # 仅处理无后缀名文件
            processed_files_count += 1
            file_sgf_count = 0

            with file_path.open('r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    # 分割ID和SGF内容
                    sgf_start_index = line.find(';')
                    if sgf_start_index != -1:
                        game_id = line[:sgf_start_index].strip()
                        sgf_content = line[sgf_start_index:].strip()
                        if not game_id or not sgf_content: continue # 过滤掉空的ID或内容

                        # 构建SGF内容
                        output_filename = f"{file_path.name}_{game_id}.sgf"
                        output_file_path = output_path / output_filename
                        full_sgf_content = f"(;GM[1]FF[4]SZ[19]AP[PythonScript]{sgf_content})"
                        output_file_path.write_text(full_sgf_content, encoding='utf-8')
                        
                        file_sgf_count += 1
                        total_sgf_count += 1
                    else:
                        print(f"  - 警告：第 {line_num} 行未找到SGF起始符号';'，跳过。")

            print(f"  - 完成，从该文件提取了 {file_sgf_count} 条记录。")

    # 7. 打印最终总结
    print("\n=============================================")
    print(f"处理完成！")
    print(f"总共处理的无后缀文件数量: {processed_files_count}")
    print(f"总共提取并生成的SGF文件数量: {total_sgf_count}")
    print(f"所有文件已保存至: {output_path.resolve()}")
    print("=============================================")


if __name__ == "__main__":
    INPUT_DIRECTORY = "GoDataset/AI/" 
    OUTPUT_DIRECTORY = "GoDataset/AI/kifu2sgf"
    process_files_to_sgf(INPUT_DIRECTORY, OUTPUT_DIRECTORY)