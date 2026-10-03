"""Selected V2: preserve V1 samples, evaluate the sky after cloud transport."""
from single_frame import SingleFrameRenderer, Renderer, compile_program, glDeleteProgram
from v2_sources import opaque_source


class SingleFrameV2Renderer(SingleFrameRenderer):
    def __init__(self,width,height):
        super().__init__(width,height)
        self.v1_sky=self.optimized_sky
        self.v2_sky=compile_program(opaque_source(self.optimized_source))

    def scene(self,**kwargs):
        super().scene(**kwargs)
        self.sky=self.v2_sky
        Renderer.scene(self,**kwargs)
        self.sky=self.original_sky

    def render(self,v2=False,optimized=False,**variant):
        # Keep V1's exact pure-sky path and small-image fallback.
        self.optimized_sky=self.v2_sky if v2 and not self.sky_only else self.v1_sky
        try:
            return super().render(optimized=optimized or v2,**variant)
        finally:
            self.optimized_sky=self.v1_sky

    def capture(self,v2=False,optimized=False,**variant):
        return super().capture(v2=v2,optimized=optimized or v2,**variant)

    def close(self):
        self.optimized_sky=self.v1_sky
        glDeleteProgram(self.v2_sky)
        super().close()
