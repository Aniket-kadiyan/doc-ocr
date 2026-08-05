"use client";

import { fitBalloonFontSize, getBalloonScreenSize } from "@/lib/balloonStyle";
import { useBalloonStyle } from "@/components/BalloonStyleProvider";

export function BalloonThumbnail({
  number,
  selected = false,
}: {
  number: number;
  selected?: boolean;
}) {
  const { style } = useBalloonStyle();
  const size = getBalloonScreenSize(style, "sidebar");
  const areaWidth = style.numberArea.width * size.width;
  const areaHeight = style.numberArea.height * size.height;
  const fontSize = fitBalloonFontSize(
    String(number),
    areaWidth,
    areaHeight,
    size.width,
    style
  );

  return (
    <span
      role="img"
      aria-label={`Balloon ${number}`}
      className="relative block shrink-0"
      style={{
        width: size.width,
        height: size.height,
        filter: selected ? "drop-shadow(0 0 3px #2563eb)" : undefined,
      }}
    >
      {/* The complete artwork is embedded in balloon-style.json. */}
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={style.artwork.dataUrl}
        alt=""
        draggable={false}
        className="absolute inset-0 h-full w-full select-none"
      />
      <span
        className="absolute flex items-center justify-center overflow-hidden text-center leading-none"
        style={{
          left: `${style.numberArea.x * 100}%`,
          top: `${style.numberArea.y * 100}%`,
          width: `${style.numberArea.width * 100}%`,
          height: `${style.numberArea.height * 100}%`,
          color: style.text.color,
          fontFamily: style.text.fontFamily,
          fontWeight: style.text.fontWeight,
          fontSize,
        }}
      >
        {number}
      </span>
    </span>
  );
}
