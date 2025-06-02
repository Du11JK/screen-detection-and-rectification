#!/usr/bin/env python3
"""
FastAPI服务器 - 提供文件上传接口，自动调用屏幕矫正处理流程
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import os
import shutil
import tempfile
from datetime import datetime
from perspective_corrector import process_screen_correction

app = FastAPI(title="屏幕矫正服务", description="上传图片文件进行屏幕检测和透视矫正")

# 配置
MODEL_PATH = './models/sam2.1_b.pt'
BASE_OUTPUT_DIR = './output'
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.heic', '.HEIC'}

def get_file_extension(filename: str) -> str:
    """获取文件扩展名"""
    return os.path.splitext(filename)[1].lower()

def is_allowed_file(filename: str) -> bool:
    """检查文件类型是否允许"""
    return get_file_extension(filename) in ALLOWED_EXTENSIONS

@app.get("/")
async def root():
    """根路径 - 服务状态"""
    return {"message": "屏幕矫正服务运行中", "status": "active"}

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    上传多个文件进行处理
    """
    if not files:
        raise HTTPException(status_code=400, detail="没有上传文件")
    
    # 检查模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=500, detail=f"模型文件不存在: {MODEL_PATH}")
    
    # 创建带时间戳的输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(BASE_OUTPUT_DIR, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for file in files:
        # 检查文件类型
        if not is_allowed_file(file.filename):
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": f"不支持的文件类型: {get_file_extension(file.filename)}"
            })
            continue
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=get_file_extension(file.filename)) as temp_file:
            try:
                # 保存上传的文件
                shutil.copyfileobj(file.file, temp_file)
                temp_file_path = temp_file.name
                
                # 调用处理流程
                process_results = process_screen_correction(
                    image_path=temp_file_path,
                    model_path=MODEL_PATH,
                    output_dir=output_dir,
                    auto_ratio=True,
                    output_size=(1920, 1080)
                )
                
                if process_results:
                    # 处理成功
                    corrected_images = process_results.get('corrected_images', {})
                    output_files = []
                    
                    for ratio, img_info in corrected_images.items():
                        if 'path' in img_info:
                            # 重命名输出文件，包含原文件名
                            original_name = os.path.splitext(file.filename)[0]
                            new_filename = f"{original_name}_corrected_{ratio}.jpg"
                            new_path = os.path.join(output_dir, new_filename)
                            
                            if os.path.exists(img_info['path']):
                                shutil.move(img_info['path'], new_path)
                                output_files.append(new_path)
                    
                    results.append({
                        "filename": file.filename,
                        "status": "success",
                        "best_ratio": process_results.get('best_ratio'),
                        "output_files": output_files,
                        "output_dir": output_dir
                    })
                else:
                    results.append({
                        "filename": file.filename,
                        "status": "error",
                        "message": "处理失败 - 未检测到屏幕或处理过程出错"
                    })
                
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": f"处理出错: {str(e)}"
                })
            
            finally:
                # 清理临时文件
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
    
    return JSONResponse(content={
        "timestamp": timestamp,
        "output_directory": output_dir,
        "processed_files": len(results),
        "results": results
    })

@app.get("/health")
async def health_check():
    """健康检查"""
    model_exists = os.path.exists(MODEL_PATH)
    return {
        "status": "healthy" if model_exists else "unhealthy",
        "model_path": MODEL_PATH,
        "model_exists": model_exists
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)