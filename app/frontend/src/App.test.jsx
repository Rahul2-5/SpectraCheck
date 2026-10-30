import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App, { DISCLAIMER, FEATURE_ONLY_NOTICE } from './App.jsx';

const health = { ready: true, feature_analysis_available: true, min_image_side: 256, max_upload_bytes: 10485760, disclaimer: DISCLAIMER };
const prediction = { label: 'AI-generated', prediction: 1, ai_probability: .72, warnings: [], spectrum_png: 'data:image/png;base64,AA==', rapsd: { radius: [1, 64, 128], log_power: [3, 2, 1] } };
const featureAnalysis = { mode: 'feature_analysis', model_available: false, label: null, prediction: null, ai_probability: null, model: null, processed_size: 256, feature_analysis: { spectral_entropy: .4567, rapsd_ring_count: 3 }, warnings: [], spectrum_png: 'data:image/png;base64,AA==', rapsd: { radius: [1, 64, 128], log_power: [3, 2, 1] } };
const response = data => ({ ok: true, json: async () => data });

describe('analysis workspace', () => {
  it('runs feature analysis when the research model is not connected', async () => {
    vi.stubGlobal('fetch', vi.fn(async url => response(url.endsWith('health') ? { ...health, ready: false } : featureAnalysis)));
    render(<App />);
    await waitFor(() => expect(screen.queryByText(/Connecting to local service/)).not.toBeInTheDocument());
    fireEvent.change(screen.getByLabelText('Choose an image'), { target: { files: [new File(['image'], 'photo.png', { type: 'image/png' })] } });
    expect(screen.getByText('photo.png')).toBeInTheDocument();
    const analyzeButton = screen.getByRole('button', { name: /analyze image/i });
    expect(analyzeButton).toBeEnabled();
    await userEvent.click(analyzeButton);
    expect(await screen.findByRole('heading', { name: 'Feature analysis only' })).toBeInTheDocument();
    expect(screen.getByText(FEATURE_ONLY_NOTICE)).toBeInTheDocument();
    expect(screen.getByText('256 × 256')).toBeInTheDocument();
    expect(screen.getByText('0.4567')).toBeInTheDocument();
    expect(screen.getByRole('img', { name: /centered log-power/i })).toBeInTheDocument();
    expect(screen.getByRole('img', { name: /radially averaged/i })).toBeInTheDocument();
    expect(screen.queryByRole('meter')).not.toBeInTheDocument();
    expect(screen.queryByText('AI-generated probability')).not.toBeInTheDocument();
  });
  it('uploads the file and displays the server probability and spectra', async () => {
    const fetcher = vi.fn(async (url) => response(url.endsWith('health') ? health : prediction));
    vi.stubGlobal('fetch', fetcher);
    render(<App />);
    await waitFor(() => expect(screen.queryByText('Connecting to local service…')).not.toBeInTheDocument());
    const file = new File(['pixels'], 'photo.png', { type: 'image/png' });
    fireEvent.change(screen.getByLabelText('Choose an image'), { target: { files: [file] } });
    await userEvent.click(screen.getByRole('button', { name: /analyze image/i }));
    expect(await screen.findByRole('heading', { name: 'AI-generated' })).toBeInTheDocument();
    expect(screen.getByRole('meter')).toHaveAttribute('aria-valuenow', '72');
    expect(fetcher.mock.calls[1][1].body).toBe(file);
    expect(screen.getByRole('img', { name: /radially averaged/i })).toBeInTheDocument();
  });
  it('rejects multiple or unsupported files and clears prior selection', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => response(health)));
    render(<App />);
    const input = screen.getByLabelText('Choose an image');
    fireEvent.change(input, { target: { files: [new File(['x'], 'photo.png', { type: 'image/png' })] } });
    fireEvent.change(input, { target: { files: [new File(['x'], 'wrong.txt', { type: 'text/plain' })] } });
    expect(screen.getByRole('alert')).toHaveTextContent('Choose a JPEG, PNG, or WebP image.');
    expect(screen.queryByText('photo.png')).not.toBeInTheDocument();
    fireEvent.drop(screen.getByText(/bring an image/i).closest('.dropzone'), { dataTransfer: { files: [new File(['a'], 'a.png'), new File(['b'], 'b.png')] } });
    expect(screen.getByRole('alert')).toHaveTextContent('Choose one image at a time.');
  });
  it('shows API errors without inventing a result and supports retry', async () => {
    vi.stubGlobal('fetch', vi.fn(async url => url.endsWith('health') ? response(health) : { ok: false, json: async () => ({ detail: 'Image smaller than target: centered crop side 100 < 256' }) }));
    render(<App />);
    await waitFor(() => expect(screen.queryByText('Connecting to local service…')).not.toBeInTheDocument());
    fireEvent.change(screen.getByLabelText('Choose an image'), { target: { files: [new File(['x'], 'tiny.png', { type: 'image/png' })] } });
    await userEvent.click(screen.getByRole('button', { name: /analyze image/i }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Image smaller than target');
    expect(screen.queryByRole('meter')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /analyze image/i })).toBeEnabled();
  });
  it('aborts a pending analysis when the image is removed', async () => {
    let pendingSignal;
    vi.stubGlobal('fetch', vi.fn(async (url, options) => {
      if (url.endsWith('health')) return response(health);
      pendingSignal = options.signal;
      return new Promise(() => {});
    }));
    render(<App />);
    await waitFor(() => expect(screen.queryByText('Connecting to local service…')).not.toBeInTheDocument());
    fireEvent.change(screen.getByLabelText('Choose an image'), { target: { files: [new File(['x'], 'photo.png', { type: 'image/png' })] } });
    await userEvent.click(screen.getByRole('button', { name: /analyze image/i }));
    await userEvent.click(screen.getByRole('button', { name: 'Remove image' }));
    expect(pendingSignal.aborted).toBe(true);
    expect(screen.queryByText('photo.png')).not.toBeInTheDocument();
    expect(screen.getByText('Awaiting image')).toBeInTheDocument();
  });
  it('gives a clear message when the service returns an HTML error page', async () => {
    vi.stubGlobal('fetch', vi.fn(async url => url.endsWith('health') ? response(health) : {
      ok: false, status: 502, json: async () => { throw new SyntaxError('Unexpected token <'); },
    }));
    render(<App />);
    await waitFor(() => expect(screen.queryByText('Connecting to local service…')).not.toBeInTheDocument());
    fireEvent.change(screen.getByLabelText('Choose an image'), { target: { files: [new File(['x'], 'image.png', { type: 'image/png' })] } });
    await userEvent.click(screen.getByRole('button', { name: /analyze image/i }));
    expect(await screen.findByRole('alert')).toHaveTextContent('The local service could not complete the analysis. Please try again.');
    expect(screen.queryByRole('meter')).not.toBeInTheDocument();
  });
});
