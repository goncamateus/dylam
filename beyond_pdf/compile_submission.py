"""Compile a submission to TMLR Beyond PDF.

Example
-------
python compile_submission.py
"""
import os
import re
from shutil import copyfile


def copy_files(src_dir, dest_dir):
  for src_fname in os.listdir(src_dir):
    try:
      os.makedirs(dest_dir, exist_ok=True)
      src_path = os.path.join(src_dir, src_fname)
      dest_path = os.path.join(dest_dir, src_fname)
      copyfile(src_path, dest_path)
      print(f'Copied {src_path} to {dest_path}')
    except:
      print(f'Issue with copying {src_path} to {dest_path}!')


def convert_math_delimiters(filepath):
  with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

  # Split into front matter and body
  fm_pattern = re.compile(r'^(---\s*\n.*?\n---\s*\n)(.*)', re.DOTALL)
  match = fm_pattern.match(content)

  if match:
    front_matter = match.group(1)
    body = match.group(2)
  else:
    front_matter = ''
    body = content

  # Only convert standalone $ to $$ in the body
  converted_body = re.sub(r'(?<![\\\$])\$(?!\$)', '$$', body)

  with open(filepath, 'w', encoding='utf-8') as f:
    f.write(front_matter + converted_body)

  print(f'Converted inline math delimiters in {filepath}')


def merge_files(submission_dir, target_dir):
  # Copy submission.md to _under_review
  dest_submission = os.path.join(target_dir, '_under_review', 'submission.md')
  copyfile(
      os.path.join(submission_dir, 'submission.md'),
      dest_submission
  )

  # Convert $...$ to $$...$$ for kramdown compatibility
  convert_math_delimiters(dest_submission)

  # Copy from assets/bibliography
  copy_files(
      os.path.join(submission_dir, 'assets/bibliography'),
      os.path.join(target_dir, 'assets/bibliography')
  )

  # Copy from assets/html
  html_base_dir = f'{submission_dir}/assets/html'
  html_target_dir = f'{target_dir}/assets/html'
  if os.path.exists(html_base_dir):
    for dir_name in os.listdir(html_base_dir):
      if dir_name == '.DS_Store':
        continue

      copy_files(
          f'{html_base_dir}/{dir_name}',
          f'{html_target_dir}/{dir_name}'
      )
  else:
    print(f'{html_base_dir} not found, skipping.')

  # Copy from assets/img
  img_base_dir = f'{submission_dir}/assets/img'
  img_target_dir = f'{target_dir}/assets/img'
  if os.path.exists(img_base_dir):
    for dir_name in os.listdir(img_base_dir):
      if dir_name == '.DS_Store':
        continue

      copy_files(
          f'{img_base_dir}/{dir_name}',
          f'{img_target_dir}/{dir_name}'
      )
  else:
    print(f'{img_base_dir} not found, skipping.')

  # Copy from assets/gif
  gif_base_dir = f'{submission_dir}/assets/gif'
  gif_target_dir = f'{target_dir}/assets/gif'
  if os.path.exists(gif_base_dir):
    for dir_name in os.listdir(gif_base_dir):
      if dir_name == '.DS_Store':
        continue

      copy_files(
          f'{gif_base_dir}/{dir_name}',
          f'{gif_target_dir}/{dir_name}'
      )
  else:
    print(f'{gif_base_dir} not found, skipping.')


def start_jekyll_server():
  os.system('cd tmlr_do_not_modify; bash ./bin/docker_run.sh')


merge_files('submission_folder', 'tmlr_do_not_modify')
start_jekyll_server()

