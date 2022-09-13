import sdl2.ext

WHITE = (256 << 16 | 256 << 8 | 256)
#WHITE = (256 << 16 | 256 << 8 | 256)

def create_sdl_surface(*shape):
    sdl2.ext.init()
    window = sdl2.ext.Window("AEStream window", shape)
    window.show()

    factory = sdl2.ext.SpriteFactory(sdl2.ext.SOFTWARE)
    renderer = factory.create_sprite_render_system(window)
    pixels = sdl2.ext.pixelaccess.pixels2d(renderer)

    return window, pixels

def events_to_bw(events):
    return events.int() * (255 << 16)