import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

export_file_url = 'https://drive.google.com/uc?export=download&id=1NRRlNEIU2_n-2B4uwbmjG6XuvQmE0ogW'
export_file_name = 'stage-1.pkl'

# classes = ["Speed limit (20km/h)","Speed limit (30km/h)", "Speed limit (50km/h)", "Speed limit (60km/h)","Speed limit (70km/h)","Speed limit (80km/h)",
# "End of speed limit (80km/h)","Speed limit (100km/h)", "Speed limit (120km/h)", "No passing", "No passing for vechiles over 3.5 metric tons" ,
# "Right-of-way at the next intersection" , "Priority road", "Yield" , "Stop" , "No vechiles", "Vechiles over 3.5 metric tons prohibited", "No entry", "General caution",
# "Dangerous curve to the left", "Dangerous curve to the right","Double curve","Bumpy road","Slippery road","Road narrows on the right"
# ,"Road work", "Traffic signals","Pedestrians","Children crossing","Bicycles crossing","Beware of ice/snow","Wild animals crossing"
# ,"End of all speed and passing limits","Turn right ahead","Turn left ahead","Ahead only","Go straight or right","Go straight or left","Keep right"
# ,"Keep left","Roundabout mandatory","End of no passing","End of no passing by vechiles over 3.5 metric tons"]
classes =  k = [str(i) for i in range(43) ]
path = Path(__file__).parent
print(path)

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
